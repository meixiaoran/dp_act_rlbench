import sys
import time


# 设置标准输出和标准错误的缓冲方式为行缓冲，这样输出会在每行结束时刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from einops import rearrange

from RLBench_ACT.act.sim_env_rlbench import make_sim_env
from diffusion_policy.common.pytorch_util import dict_apply
import os

import hydra  # 用于配置管理和依赖注入
import torch  # 用于加载和操作 PyTorch 模型
import dill  # 用于 pickle 操作（比标准 pickle 支持更多对象类型）

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 导入基类，代表工作空间

from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict)

from RLBench.rlbench.const import SUPPORTED_ROBOTS

import numpy as np
from scipy.spatial.transform import Rotation
import random

def set_seed(seed=42):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # Numpy 随机种子
    torch.manual_seed(seed)  # PyTorch CPU 随机种子

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 仅用于当前 GPU
        torch.cuda.manual_seed_all(seed)  # 用于所有 GPU
        torch.backends.cudnn.deterministic = True  # 保持确定性（影响某些操作）
set_seed(42)

def compute_pose_in_world(T_BW, T_AB):
    # 从 T_BW 和 T_AB 中提取位置和四元数
    p_BW = np.array(T_BW[:3])  # B 在世界坐标系中的位置
    q_BW = np.array(T_BW[3:])  # B 在世界坐标系中的四元数

    p_AB = np.array(T_AB[:3])  # A 在 B 坐标系中的位置
    q_AB = np.array(T_AB[3:])  # A 在 B 坐标系中的四元数

    # 计算 A 在世界坐标系中的位置
    # 将四元数转换为旋转矩阵
    R_BW = Rotation.from_quat(q_BW).as_matrix()  # 从四元数生成旋转矩阵
    p_AW = R_BW @ p_AB + p_BW  # 世界坐标系中的位置

    # 计算 A 在世界坐标系中的姿态
    q_AW = Rotation.from_quat(q_BW) * Rotation.from_quat(q_AB)  # 四元数乘法
    q_AW = q_AW.as_quat()  # 转换为四元数格式

    # 返回结果：位置和四元数组合成 7 元素数组
    return np.concatenate([p_AW, q_AW])

def get_image(ts):
    curr_images = []

    # 获取 wrist_rgb 图像

    wrist_rgb = ts.wrist_rgb
    curr_image = rearrange(wrist_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)

    # 获取 head_rgb 图像

    head_rgb = ts.head_rgb
    curr_image = rearrange(head_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)

    # 组合所有的图像
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def main(checkpoint, ckpt_name0, device, task_name, robot_name, epochs=50, onscreen_render=False, variation=0, save_episode=True):

    # 检查输出目录是否存在，若存在则询问是否覆盖
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    #
    # # 创建输出目录
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型检查点文件
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)  # 使用 dill 读取 pickle 文件
    cfg = payload['cfg']  # 从加载的 payload 中提取配置
    print(dir(cfg))
    cls = hydra.utils.get_class(cfg._target_)  # 获取配置中指定的类
    workspace = cls(cfg)  # 实例化工作空间对象
    workspace: BaseWorkspace  # 确保 workspace 是 BaseWorkspace 类型
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # 加载模型的 payload 数据到工作空间中

    print("模型：")
    model_state_dict = payload['state_dicts']['model']
    print(model_state_dict.keys())
    policy = workspace.model
    print(policy.model)
    # policy.load_state_dict(model_state_dict, strict=True)

    if cfg.training.use_ema:  # 如果使用了 EMA（Exponential Moving Average），则使用 EMA 模型
        policy = workspace.ema_model

    # 将模型移到指定的设备（GPU 或 CPU）
    device = torch.device(device)
    policy.eval().to(device)  # 设置模型为评估模式，并转移到 GPU

    policy.num_inference_steps = 100  # 设置 DDIM 推理步骤数
    policy.n_action_steps = 8  # 设置动作步骤数

    print("n_action_steps", policy.n_action_steps)
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps  # 获取观察步数
    print("n_obs_steps: ", n_obs_steps)  # 打印观察步数
    print("obs_res:", obs_res)
    print("robot_name:", robot_name)
    env = make_sim_env(task_name, onscreen_render, robot_name)  # 创建模拟环境
    env_max_reward = 1  # 设置最大奖励（模拟环境）
    env.set_variation(0)
    episode_returns = []  # 记录每回合的总奖励
    highest_rewards = []  # 记录每回合的最大奖励
    robot_setup: str = 'panda'
    robot_setup = robot_setup.lower()
    arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
        robot_setup]
    arm, gripper = arm_class(), gripper_class()
    arm_pose = arm.get_pose()



    ckpt_dir = os.path.splitext(checkpoint)[0]
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(epochs):
        start_time = time.time()
        num_rollouts = epochs
        rollout_id = epoch
        # 设置环境的任务变种
        # if variation >= 0:
        #     env.set_variation(variation)  # 使用指定的变种
        # else:
        #     random_variation = np.random.randint(3)  # 随机选择一个变种
        #     env.set_variation(random_variation)
        env.set_variation(variation)

        try:
            descriptions, ts_obs = env.reset()
            time.sleep(1)  # 等待1000秒
        except Exception as e:
            print(f"An error occurred: {e}")

        # print(dir(ts_obs))


        ### 评估循环
        image_list = []  # 用于可视化图像的列表
        qpos_list = []  # 记录关节位置的列表
        target_qpos_list = []  # 记录目标关节位置的列表
        rewards = []  # 记录奖励的列表
        t = 0  # 时间步计数器

        with torch.no_grad():

            policy.reset()
            import numpy as np
            terminate = False
            obs = ts_obs
            joint_positions = np.append(obs.gripper_pose, obs.gripper_open)


            while not terminate:
                obs = ts_obs  # 获取当前观察
                image_list.append({'front': obs.front_rgb, 'wrist': obs.wrist_rgb})  # 保存图像
                import numpy as np

                qpos_list.append({'qpos': np.append(obs.gripper_pose, obs.gripper_open)})
                # 获取所有包含 'front' 键的字典数量


                # 假设你已经有了一个空字典来存储图像
                import numpy as np

                # 假设你已经有了一个空字典来存储图像
                # 在初始化时指定形状，确保拼接时的形状匹配
                observation = {
                    'wrist': np.empty((0, obs.wrist_rgb.shape[0], obs.wrist_rgb.shape[1], obs.wrist_rgb.shape[2]),
                                      dtype=np.uint8),
                    # 假设 wrist_rgb 是 (height, width, channels)，直接设定 dtype 为 uint8
                    'front': np.empty((0, obs.front_rgb.shape[0], obs.front_rgb.shape[1], obs.front_rgb.shape[2]),
                                     dtype=np.uint8),

                    # # 如果你想创建一个与 joint_positions 相同维度的空数组
                    'qpos': np.empty((0,) + joint_positions.shape, dtype=np.float32),
                    # # 假设 front_rgb 也是 (height, width, channels)，直接设定 dtype 为 uint8
                }

                if t == 0:
                    # 直接在拼接时将图像转换为 uint8 类型
                    observation["front"] = np.concatenate(
                        [observation["front"], obs.front_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    observation["wrist"] = np.concatenate(
                        [observation["wrist"], obs.wrist_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)

                    joint_positions1 = np.append(obs.gripper_pose, obs.gripper_open)
                    observation["qpos"] = np.concatenate([observation["qpos"], joint_positions1[np.newaxis, ...]], axis=0)

                    # 直接在拼接时将图像转换为 uint8 类型
                    observation["front"] = np.concatenate(
                        [observation["front"], obs.front_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    observation["wrist"] = np.concatenate(
                        [observation["wrist"], obs.wrist_rgb[np.newaxis, ...].astype(np.uint8)], axis=0)
                    joint_positions2 = np.append(obs.gripper_pose, obs.gripper_open)
                    observation["qpos"] = np.concatenate([observation["qpos"], joint_positions2[np.newaxis, ...]], axis=0)

                else:
                    last_image = image_list[-2]  # 倒数第一个元素
                    second_last_image = image_list[-3]  # 倒数第二个元素

                    last_qpos = qpos_list[-2]
                    second_last_qpos = qpos_list[-3]
                    # print("last_qpos", last_qpos)
                    # print("second_qpos", second_last_qpos)
                    observation['wrist'] = np.concatenate(
                        [observation['wrist'], second_last_image['wrist'][np.newaxis, ...]], axis=0)
                    observation['wrist'] = np.concatenate([observation['wrist'], last_image['wrist'][np.newaxis, ...]],
                                                         axis=0)


                    observation['front'] = np.concatenate(
                        [observation['front'], second_last_image['front'][np.newaxis, ...]], axis=0)
                    observation['front'] = np.concatenate([observation['front'], last_image['front'][np.newaxis, ...]],
                                                        axis=0)

                    observation['qpos'] = np.concatenate([observation['qpos'], second_last_qpos['qpos'][np.newaxis, ...]], axis=0)
                    observation['qpos'] = np.concatenate([observation['qpos'], last_qpos['qpos'][np.newaxis, ...]],axis=0)


                # 将环境的观察数据（`obs`）转换为模型所需要的输入格式
                obs_dict_np = get_real_obs_dict(
                    env_obs=observation, shape_meta=cfg.task.shape_meta)

                # print("qpos", obs_dict_np["qpos"])


                # 将环境的原始观察数据（`obs`）转换为符合模型要求的字典格式，形状元数据从配置中提取。
                # 使用字典映射，将观察数据转换为 PyTorch 张量，并将其移到指定的设备（GPU 或 CPU）。
                obs_dict = dict_apply(obs_dict_np,
                                      lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                # `dict_apply` 函数用于将 numpy 数组转换为张量，并添加批量维度。



                # 使用策略模型推理动作
                # print("model start eval")
                result = policy.predict_action(obs_dict)
                # 使用当前的观察数据输入到策略模型中，返回动作结果。

                action = result['action_pred'][0].detach().to('cpu').numpy()
                # print("action", action)
                # 获取模型输出的动作数据，`.detach()` 用于移除计算图，`to('cpu')` 将张量转移到 CPU，
                # `numpy()` 用于转换为 NumPy 数组，方便后续处理。
                # print(action)
                # print(action)
                for i in range(10):
                    action_first_six = action[i]

                    quaternion_raw = action_first_six[3:7]
                    # 计算平方和的平方根
                    norm = np.sqrt(np.sum(quaternion_raw ** 2))
                    # 如果 norm 为 0，避免除以 0
                    if norm > 0:
                        quaternion_normalized = quaternion_raw / norm
                    else:
                        # 如果 norm 为 0，设置为单位四元数
                        quaternion_normalized = np.array([1, 0, 0, 0])
                    # 替换到原数组中
                    action_first_six[3:7] = quaternion_normalized
                    action_world=action_first_six

                    # 坐标系转换
                    # action_world = compute_pose_in_world(arm_pose, action_first_six[:-1])
                    # action_world = np.concatenate([action_world, np.array([action_first_six[-1]])])

                    # print("action_world", action_world)
                # 发送动作到环境并获取新的状态和奖励
                #     if action_first_six[6] < 0.1:
                #         action_first_six[6] = 1.0
                    try:
                        ts_obs, reward, terminate = env.step(action_world)
                    except Exception as e:
                        print("action failed")
                        print(action_world)
                        # print("timeout:" + str(time.time() - start_time))
                        if(time.time() - start_time > 50.0):
                            t = 500
                            break
                        continue

                    obs = ts_obs
                    image_list.append(
                        {'front': obs.front_rgb, 'wrist': obs.wrist_rgb})  # 保存图像
                    qpos_list.append({'qpos': np.append(obs.gripper_pose, obs.gripper_open)})
                    rewards.append(reward)  # 记录奖励

                    t = t + 1  # 增加时间步计数器

                    # if reward - env_max_reward >= 0.0 and action_world[7] == 1.0:
                    if reward - env_max_reward >= 0.0:
                        print("griper:" + str(action_world[7]))
                        terminate = True
                        break  # 如果获得最大奖励，直接跳出循环（成功完成任务)

                    # print(t)


                if t >= 500:
                    rewards = [0 for _ in rewards]
                    break

        # 记录每个回合的奖励信息
        rewards = np.array(rewards)  # 将奖励列表转换为NumPy数组
        episode_return = np.sum(rewards[rewards != None])  # 计算本回合的总奖励，忽略 None 值
        episode_returns.append(episode_return)  # 将本回合的总奖励添加到 episode_returns 列表中

        # 计算本回合的最大奖励
        episode_highest_reward = np.max(rewards)  # 获取本回合的最大奖励
        highest_rewards.append(episode_highest_reward)  # 将最大奖励添加到 highest_rewards 列表中

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
            print("t", t+1)
        # 返回成功率和平均回报
    return success_rate, avg_return


# %%
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run diffusion policy with command-line arguments.")
    parser.add_argument('--checkpoint', type=str, required=True, default='/home/mar/diffusion_policy/data/pcik_up_cup_1800_dp-t.ckpt ')
    parser.add_argument('--task_name', type=str, required=True, default='pick_up_cup')

    # 也可以为其他参数添加默认值
    parser.add_argument('--ckpt_name0', type=str, default='latest.ckpt')
    parser.add_argument('--robot_name', type=str, default='panda')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--no_onscreen_render', dest='onscreen_render', action='store_false')
    parser.set_defaults(onscreen_render=True)
    parser.add_argument('--variation', type=int, default=0)
    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        ckpt_name0=args.ckpt_name0,
        device=args.device,
        task_name=args.task_name,
        robot_name=args.robot_name,
        epochs=args.epochs,
        onscreen_render=args.onscreen_render,
        variation=args.variation
    )