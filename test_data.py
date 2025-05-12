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

import h5py

# 假设数据存储在 demo_0.h5 文件中
file_path = '/home/mar/diffusion_policy/data/dataset/reach_target(1)/merged_data.hdf5'

task_name = "reach_target"
onscreen_render = True
robot_name = "ur5"


env = make_sim_env(task_name, onscreen_render, robot_name)  # 创建模拟环境
descriptions, ts_obs = env.reset()  # 重置环境，获取初始观测
with h5py.File(file_path, 'r') as f:
    actions = f["data"]["demo_0"]['action'][:]  # 获取 action 数据集

    # 使用 for 循环遍历每个 action
    for idx, action in enumerate(actions):
        ts_obs, reward, terminate = env.step(action)
        print(f"Action {idx+1}: {action}")
        # 在这里处理每个 action

