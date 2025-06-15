import sys
from calendar import EPOCH


# 设置标准输出和标准错误的缓冲方式为行缓冲，这样输出会在每行结束时刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import matplotlib.pyplot as plt
from einops import rearrange
import argparse  # 确保导入了 argparse
from RLBench_ACT.act.sim_env_rlbench import make_sim_env
from diffusion_policy.common.pytorch_util import dict_apply
import os
import pathlib
import click  # 用于命令行参数解析
import hydra  # 用于配置管理和依赖注入
import torch  # 用于加载和操作 PyTorch 模型
import dill  # 用于 pickle 操作（比标准 pickle 支持更多对象类型）
import wandb  # 用于日志和监控
import json  # 用于处理 JSON 数据
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 导入基类，代表工作空间
import click
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict)
import numpy as np
from RLBench_ACT.act.constants import DT

@click.command()  # 使用 click 库定义一个命令行工具
@click.option('-c', '--checkpoint', required=True)  # 检查点路径参数，必须提供
# @click.option('-o', '--output_dir', required=True)  # 输出目录参数，必须提供
@click.option('-d', '--device', default='cuda:0')  # 设备参数，默认为 'cuda:0'，即使用 GPU
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

    cls = hydra.utils.get_class(cfg._target_)  # 获取配置中指定的类
    workspace = cls(cfg)  # 实例化工作空间对象
    workspace: BaseWorkspace  # 确保 workspace 是 BaseWorkspace 类型
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # 加载模型的 payload 数据到工作空间中

    policy = workspace.model
    if cfg.training.use_ema:  # 如果使用了 EMA（Exponential Moving Average），则使用 EMA 模型
        policy = workspace.ema_model

    # 将模型移到指定的设备（GPU 或 CPU）
    device = torch.device(device)
    policy.to(device)
    policy.eval()  # 设置模型为评估模式（关闭 Dropout 等）

    policy.num_inference_steps = 16  # 设置 DDIM 推理步骤数
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1  # 设置动作步骤数

    print("n_action_steps", policy.n_action_steps)
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps  # 获取观察步数
    print("n_obs_steps: ", n_obs_steps)  # 打印观察步数
    print("obs_res:", obs_res)

    env = make_sim_env(task_name, onscreen_render, robot_name)  # 创建模拟环境
    env_max_reward = 1  # 设置最大奖励（模拟环境）
    episode_returns = []  # 记录每回合的总奖励
    highest_rewards = []  # 记录每回合的最大奖励

    max_timesteps = 200

    # 定义预处理和后处理函数
    ckpt_dir = "/home/mar/diffusion_policy/data/outputs/2024.12.04/22.07.02_train_diffusion_transformer_hybrid_reach_target/checkpoints"


    for epoch in range(epochs):
        num_rollouts = epochs
        rollout_id = epoch
        # 设置环境的任务变种
        if variation >= 0:
            env.set_variation(variation)  # 使用指定的变种
        else:
            random_variation = np.random.randint(3)  # 随机选择一个变种
            env.set_variation(random_variation)

        descriptions, ts_obs = env.reset()  # 重置环境，获取初始观测
        print(dir(ts_obs))

        ### 评估循环
        image_list = []  # 用于可视化图像的列表
        qpos_list = []  # 记录关节位置的列表
        target_qpos_list = []  # 记录目标关节位置的列表
        rewards = []  # 记录奖励的列表
        t = 0  # 时间步计数器

        with torch.no_grad():
            # 在推理阶段，禁用梯度计算以节省内存和计算资源。
            policy.reset()
            # 重置策略网络（如果需要的话），例如重置内部状态或模型参数。
            # import numpy as np
            #
            # # 假设你有四张形状为 (128, 128, 3) 的 RGB 图像
            # image1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            # image2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            # image3 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            # image4 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            #
            # # 将这些图像放入一个列表
            # images = [image1, image2, image3, image4]
            #
            # # 将四张图像堆叠为 (4, 128, 128, 3) 的形状
            # images_stack = np.stack(images, axis=0)
            #
            # # 现在的形状是 (4, 128, 128, 3)，我们希望将其重塑为 (2, 128, 128, 3) 这种格式
            # # 这里 n_obs_steps = 2，意味着我们有 2 张图像分别对应 'wrist' 和 'head'
            #
            # # 创建字典并分配
            # obs = {
            #     'wrist': images_stack[:2],  # 选择前两张图像作为 'wrist'
            #     'head': images_stack[2:],  # 选择后两张图像作为 'head'
            # }
            #
            # # 查看字典中的数据形状
            # print("obs_dict['wrist'].shape:", obs['wrist'].shape)  # 输出: (2, 128, 128, 3)
            # print("obs_dict['head'].shape:", obs['head'].shape)  # 输出: (2, 128, 128, 3)

            obs = ts_obs  # 获取当前观察
            image_list.append({'front': obs.front_rgb, 'head': obs.head_rgb, 'wrist': obs.wrist_rgb})  # 保存图像
            # 打印类型

            # 假设 obs 是一个对象，包含 'head_rgb' 和 'wrist_rgb' 属性
            # 创建一个字典，将 'head_rgb' 和 'wrist_rgb' 映射为字典中的键值对
            import numpy as np

            # 假设你已经有了一个空字典来存储图像
            import numpy as np

            # 假设你已经有了一个空字典来存储图像
            # 在初始化时指定形状，确保拼接时的形状匹配
            rgb_images = {
                'wrist': np.empty((0, obs.wrist_rgb.shape[0], obs.wrist_rgb.shape[1], obs.wrist_rgb.shape[2])),
                # 假设 wrist_rgb 是 (height, width, channels)
                'head': np.empty((0, obs.head_rgb.shape[0], obs.head_rgb.shape[1], obs.head_rgb.shape[2]))
                # 假设 head_rgb 也是 (height, width, channels)
            }

            # 拼接新的图像
            rgb_images["wrist"] = np.concatenate([rgb_images["wrist"], obs.wrist_rgb[np.newaxis, ...]], axis=0)
            rgb_images["head"] = np.concatenate([rgb_images["head"], obs.head_rgb[np.newaxis, ...]], axis=0)

            # 拼接新的图像
            rgb_images["wrist"] = np.concatenate([rgb_images["wrist"], obs.wrist_rgb[np.newaxis, ...]], axis=0)
            rgb_images["head"] = np.concatenate([rgb_images["head"], obs.head_rgb[np.newaxis, ...]], axis=0)

            # 打印图像的形状
            print(rgb_images["wrist"].shape)  # 输出 wrist 的形状
            print(rgb_images["head"].shape)  # 输出 head 的形状

            # 将环境的观察数据（`obs`）转换为模型所需要的输入格式
            obs_dict_np = get_real_obs_dict(
                env_obs=rgb_images, shape_meta=cfg.task.shape_meta)
            # 将环境的原始观察数据（`obs`）转换为符合模型要求的字典格式，形状元数据从配置中提取。

            # 使用字典映射，将观察数据转换为 PyTorch 张量，并将其移到指定的设备（GPU 或 CPU）。
            obs_dict = dict_apply(obs_dict_np,
                                  lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            # `dict_apply` 函数用于将 numpy 数组转换为张量，并添加批量维度。

            # 使用策略模型推理动作
            result = policy.predict_action(obs_dict)
            # 使用当前的观察数据输入到策略模型中，返回动作结果。

            action = result['action'][0].detach().to('cpu').numpy()
            # 获取模型输出的动作数据，`.detach()` 用于移除计算图，`to('cpu')` 将张量转移到 CPU，
            # `numpy()` 用于转换为 NumPy 数组，方便后续处理。
            print("action: ", action)
            for i in range(9):
                action_first_six = action[i]
                # 发送动作到环境并获取新的状态和奖励
                ts_obs, reward, terminate = env.step(action_first_six)

            rewards.append(reward)  # 记录奖励

            if reward == env_max_reward:
                break  # 如果获得最大奖励，直接跳出循环（成功完成任务

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
            # if (epoch):
            #     if save_episode:  # 如果设置了保存回合视频
            #         # 保存回合的视频文件，文件名包含模型检查点名、回合ID和是否成功完成任务的标记
            #         save_videos(image_list, DT, video_path=os.path.join(ckpt_dir,
            #                                                             f'video_{ckpt_name0}_{epoch}_{episode_highest_reward == env_max_reward}.mp4'))

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


# %%
if __name__ == '__main__':
    ckpt_dir = "/home/mar/diffusion_policy/data/outputs/2024.12.04/22.07.02_train_diffusion_transformer_hybrid_reach_target/checkpoints/latest.ckpt"
    checkpoint = ckpt_dir
    ckpt_name0 = "latest.ckpt"
    task_name = "reach_target"
    robot_name = "ur5"
    device = "cuda:0"
    main(checkpoint, ckpt_name0, device, task_name, robot_name, epochs=50, onscreen_render=True, variation=0)
