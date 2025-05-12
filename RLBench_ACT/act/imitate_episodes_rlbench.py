import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

import IPython
e = IPython.embed
# 用于训练act
def main(args):
    # 设置NumPy的输出格式，使其在打印时一行显示更多内容
    np.set_printoptions(linewidth=300)

    # 设置随机种子，确保实验可复现
    set_seed(1)

    # 解析命令行参数
    is_eval = args['eval']  # 是否进行评估
    ckpt_dir = args['ckpt_dir']  # 保存/加载模型检查点的目录
    policy_class = args['policy_class']  # 使用的策略类
    onscreen_render = args['onscreen_render']  # 是否进行屏幕渲染
    task_name = args['task_name']  # 任务名称
    batch_size_train = args['batch_size']  # 训练时的批大小
    batch_size_val = args['batch_size']  # 验证时的批大小
    num_epochs = args['num_epochs']  # 总训练轮数
    robot_name = args['robot']  # 机器人名称（例如Panda或Sawyer）

    # 设置任务类型为仿真任务
    is_sim = True
    if is_sim:
        from constants import SIM_TASK_CONFIGS  # 从常量文件中导入仿真任务配置
        task_config = SIM_TASK_CONFIGS[task_name]  # 获取对应任务的配置

    # 获取任务配置中的参数
    dataset_dir = task_config['dataset_dir']  # 数据集目录
    num_episodes = task_config['num_episodes'] * task_config['num_variation']  # 计算总的实验集数
    print(f"{task_config['num_episodes']=}, {task_config['num_variation']=}, {num_episodes=}, ")
    episode_len = task_config['episode_len']  # 每个实验的长度
    camera_names = task_config['camera_names']  # 任务中使用的摄像头名称

    # 根据不同的机器人设置状态维度
    if robot_name == "panda" or robot_name == "sawyer":
        state_dim = 8  # 对于panda或sawyer，状态维度是8
    else:
        state_dim = 7  # 其他机器人，状态维度是7

    # 设置学习率和骨干网络的配置
    lr_backbone = 1e-5  # 骨干网络的学习率
    backbone = 'resnet18'  # 使用的骨干网络是ResNet-18

    # 根据策略类配置不同的策略参数
    if 'ACT' in policy_class:  # 如果策略是ACT类型
        enc_layers = 4  # 编码器层数
        dec_layers = 7  # 解码器层数
        nheads = 8  # 多头注意力的数量
        policy_config = {'lr': args['lr'],  # 策略的学习率
                         'num_queries': args['chunk_size'],  # 每个查询的大小
                         'kl_weight': args['kl_weight'],  # KL散度的权重
                         'hidden_dim': args['hidden_dim'],  # 隐藏层的维度
                         'dim_feedforward': args['dim_feedforward'],  # 前馈神经网络的维度
                         'lr_backbone': lr_backbone,  # 骨干网络学习率
                         'backbone': backbone,  # 骨干网络
                         'enc_layers': enc_layers,  # 编码器层数
                         'dec_layers': dec_layers,  # 解码器层数
                         'nheads': nheads,  # 多头注意力的数量
                         'camera_names': camera_names,  # 摄像头名称
                         'state_dim': state_dim,  # 状态维度
                         }
    elif policy_class == 'CNNMLP':  # 如果策略是CNNMLP类型
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim, }
    else:
        raise NotImplementedError  # 如果策略类不在已定义的选项中，抛出错误

    # 配置训练的其他超参数和环境设置
    config = {
        'num_epochs': num_epochs,  # 总训练轮数
        'ckpt_dir': ckpt_dir,  # 模型保存目录
        'episode_len': episode_len,  # 每个实验的长度
        'state_dim': state_dim,  # 状态维度
        'lr': args['lr'],  # 学习率
        'policy_class': policy_class,  # 策略类
        'onscreen_render': onscreen_render,  # 是否进行屏幕渲染
        'policy_config': policy_config,  # 策略的配置
        'task_name': task_name,  # 任务名称
        'seed': args['seed'],  # 随机种子
        'temporal_agg': args['temporal_agg'],  # 是否使用时间聚合
        'camera_names': camera_names,  # 摄像头名称
        'real_robot': not is_sim,  # 是否使用真实机器人（非仿真）
        'robot_name': robot_name,  # 机器人名称
    }

    # 如果是评估模式
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']  # 评估时加载最佳模型
        results = []
        for ckpt_name in ckpt_names:
            # 进行策略评估，返回成功率和平均返回
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        # 输出评估结果
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()  # 完成评估后退出程序

    # 加载训练和验证数据
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train,
                                                           batch_size_val)

    # 保存数据集的统计信息
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)  # 如果目录不存在，创建目录
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')  # 设置数据集统计信息的保存路径
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)  # 将统计信息保存为pickle文件

    # 训练策略，并获取最好的模型检查点信息
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info  # 获取最佳模型的epoch、验证损失和模型权重

    # 保存最好的模型检查点
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')  # 设置保存路径
    torch.save(best_state_dict, ckpt_path)  # 保存模型状态字典
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')  # 输出最佳模型的信息


def make_policy(policy_class, policy_config):
    if 'ACT' in policy_class:
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if 'ACT' in policy_class:
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_image(ts, camera_names, policy_class):  # for RLBench
    curr_images = []
    wrist_rgb = ts.wrist_rgb
    curr_image = rearrange(wrist_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)

    if "head" in camera_names:
        head_rgb = ts.head_rgb
        curr_image = rearrange(head_rgb, 'h w c -> c h w')
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True, num_verification=50, variation=0):
    # 固定随机种子，确保结果可复现
    set_seed(10)

    # 从配置中提取相关参数
    ckpt_dir = config['ckpt_dir']  # 模型检查点目录
    state_dim = config['state_dim']  # 状态空间维度
    real_robot = config['real_robot']  # 是否使用真实机器人
    policy_class = config['policy_class']  # 策略类型（例如ACT，CNNMLP等）
    onscreen_render = config['onscreen_render']  # 是否启用屏幕渲染
    policy_config = config['policy_config']  # 策略的配置参数
    camera_names = config['camera_names']  # 相机的名称列表
    max_timesteps = config['episode_len']  # 每个回合的最大时间步数
    task_name = config['task_name']  # 任务名称
    temporal_agg = config['temporal_agg']  # 是否启用时间序列聚合（如动作序列的平滑）
    robot_name = config['robot_name']  # 机器人名称

    # 加载训练好的策略和统计数据
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)  # 检查点文件路径
    policy = make_policy(policy_class, policy_config)  # 创建策略对象
    loading_status = policy.load_state_dict(torch.load(ckpt_path))  # 加载策略权重
    ckpt_name0 = ckpt_name.split('.')[0]  # 获取去除扩展名的检查点名称
    print(loading_status)  # 打印加载状态
    policy.cuda()  # 将策略模型转移到GPU
    policy.eval()  # 将策略设置为评估模式（不计算梯度等）
    print(f'Loaded: {ckpt_path}')  # 打印加载的模型路径

    # 加载统计数据（如动作均值和标准差）
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 定义预处理和后处理函数
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # 加载环境（如果是模拟环境）
    if not real_robot:
        from sim_env_rlbench import make_sim_env
        env = make_sim_env(task_name, onscreen_render, robot_name)  # 创建模拟环境
        env_max_reward = 1  # 设置最大奖励（模拟环境）

    query_frequency = policy_config['num_queries']  # 策略查询频率
    if temporal_agg:
        query_frequency = 1  # 如果启用了时间序列聚合，则查询频率为1
        num_queries = policy_config['num_queries']  # 动作序列的长度

    # 增加最大时间步数的一些冗余，以适应真实机器人任务
    max_timesteps = int(max_timesteps * 1.3)

    num_rollouts = num_verification  # 总共进行的回合数
    episode_returns = []  # 记录每回合的总奖励
    highest_rewards = []  # 记录每回合的最大奖励

    # 多回合验证
    for rollout_id in range(num_rollouts):
        gripper_flag = 0  # 是否使用夹爪（假设是0为未使用）

        # 设置环境的任务变种
        if variation >= 0:
            env.set_variation(variation)  # 使用指定的变种
        else:
            random_variation = np.random.randint(3)  # 随机选择一个变种
            env.set_variation(random_variation)

        descriptions, ts_obs = env.reset()  # 重置环境，获取初始观测

        ### 评估循环
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()  # 记录关节位置历史
        image_list = []  # 用于可视化图像的列表
        qpos_list = []  # 记录关节位置的列表
        target_qpos_list = []  # 记录目标关节位置的列表
        rewards = []  # 记录奖励的列表

        with torch.inference_mode():  # 关闭梯度计算，提高推理速度
            path = []  # 存储路径的列表
            t = 0  # 时间步计数器
            for timestep in range(max_timesteps):
                obs = ts_obs  # 获取当前观察
                image_list.append({'front': obs.front_rgb, 'head': obs.head_rgb, 'wrist': obs.wrist_rgb})  # 保存图像

                # 将关节位置和夹爪信息合并为一个状态
                qpos_numpy = np.array(np.append(obs.joint_positions, obs.gripper_open))  # 6 + 1 = 8维状态

                qpos = pre_process(qpos_numpy)  # 对状态进行预处理
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # 转换为张量并送入GPU
                qpos_history[:, t] = qpos  # 记录当前时间步的状态

                curr_image = get_image(obs, camera_names, policy_class)  # 获取当前图像信息

                ### 查询策略（获取动作）
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)  # 获取动作

                    if temporal_agg:  # 如果启用了时间序列聚合
                        # 对历史动作进行聚合
                        all_time_actions[t, t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]

                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  # 去除无效动作
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        k = 0.01  # 指数加权系数
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))  # 计算加权
                        exp_weights = exp_weights / exp_weights.sum()  # 归一化权重
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)  # 转换为张量

                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)  # 聚合后的动作
                    else:
                        raw_action = all_actions[:, t % query_frequency]  # 不使用聚合，直接选择当前动作
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)  # 使用CNNMLP策略获取动作
                else:
                    raise NotImplementedError  # 如果策略类型不支持，抛出异常

                ### 后处理动作（还原为实际的动作）
                raw_action = raw_action.squeeze(0).cpu().numpy()  # 去掉批次维度并转移回CPU
                action = post_process(raw_action)  # 后处理动作

                # 发送动作到环境并获取新的状态和奖励
                ts_obs, reward, terminate = env.step(action)
                qpos_list.append(qpos_numpy)  # 记录当前的关节位置
                rewards.append(reward)  # 记录奖励

                if reward == env_max_reward:
                    break  # 如果获得最大奖励，直接跳出循环（成功完成任务）

                t = t + 1  # 增加时间步计数器

            plt.close()  # 关闭图形窗口（如果有可视化的话）

        # 记录每个回合的奖励信息
        rewards = np.array(rewards)  # 将奖励列表转换为NumPy数组
        episode_return = np.sum(rewards[rewards != None])  # 计算本回合的总奖励，忽略 None 值
        episode_returns.append(episode_return)  # 将本回合的总奖励添加到 episode_returns 列表中

        # 计算本回合的最大奖励
        episode_highest_reward = np.max(rewards)  # 获取本回合的最大奖励
        highest_rewards.append(episode_highest_reward)  # 将最大奖励添加到 highest_rewards 列表中

        # 打印当前回合的评估信息（这部分被注释掉了）
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        # print(f'{rollout_id} Rollout with {t} steps for [{descriptions[0]}]: {episode_highest_reward==env_max_reward}')

        # 如果是非首轮（rollout_id != 0），且需要保存视频
        if (rollout_id):
            if save_episode:  # 如果设置了保存回合视频
                # 保存回合的视频文件，文件名包含模型检查点名、回合ID和是否成功完成任务的标记
                save_videos(image_list, DT, video_path=os.path.join(ckpt_dir,
                                                                    f'video_{ckpt_name0}_{rollout_id}_{episode_highest_reward == env_max_reward}.mp4'))

    # 计算成功率：统计最大奖励等于环境的最大奖励的比例
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)

    # 计算平均回报：所有回合的总奖励的平均值
    avg_return = np.mean(episode_returns)

    # 构建摘要字符串，输出成功率和平均回报
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'

    # 针对每个奖励值，从0到env_max_reward，统计获得至少该奖励值的回合数和比例
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()  # 获得至少r奖励的回合数
        more_or_equal_r_rate = more_or_equal_r / num_rollouts  # 获得至少r奖励的回合比例
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

    # 打印回合的统计摘要
    print(summary_str)

    # 保存成功率和相关统计信息到文本文件
    result_file_name = 'result_' + ckpt_name0 + f'({more_or_equal_r_rate * 100}%).txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)  # 写入统计摘要
        # f.write(repr(episode_returns))  # 如果需要，可以保存所有回合的奖励列表（这行代码被注释掉了）
        f.write('\n\n')
        f.write(repr(highest_rewards))  # 保存最大奖励列表

    # 返回成功率和平均回报
    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        # print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        # print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        # print(f'{train_dataloader=}')
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        # print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # 保存训练过程的曲线图
    for key in train_history[0]:  # 遍历训练历史记录的每个指标
        # 设置保存路径，图表文件名包含训练指标的名称、训练轮数和随机种子
        plot_path = os.path.join(ckpt_dir, f'trainval_{key}_epoch_seed{seed}.png')

        # 创建一个新的图形对象
        plt.figure()

        # 提取训练历史中每个epoch对应的当前指标值
        train_values = [summary[key].item() for summary in train_history]
        # 提取验证历史中每个epoch对应的当前指标值
        val_values = [summary[key].item() for summary in validation_history]

        # 绘制训练集和验证集的曲线
        # np.linspace(0, num_epochs-1, len(train_history)) 生成从0到num_epochs-1的均匀分布的点，用于x轴
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')  # 绘制训练集的曲线
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')  # 绘制验证集的曲线

        # 可以选择设置y轴的范围，这里注释掉了
        # plt.ylim([-0.1, 1])

        # 自动调整布局，以防图形标签重叠
        plt.tight_layout()

        # 显示图例，区分训练集和验证集的曲线
        plt.legend()

        # 设置标题为当前绘制的指标名称
        plt.title(key)

        # 将绘制的图像保存为PNG文件
        plt.savefig(plot_path)

    # 输出保存路径，告知用户图表已保存
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--robot', action='store', default='sawyer', type=str, help='which robot you want to use')

    main(vars(parser.parse_args()))
