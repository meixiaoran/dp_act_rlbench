import sys
# 设置标准输出和标准错误的缓冲方式为行缓冲，这样输出会在每行结束时刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

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
@click.command()  # 使用 click 库定义一个命令行工具
@click.option('-c', '--checkpoint', required=True)  # 检查点路径参数，必须提供
# @click.option('-o', '--output_dir', required=True)  # 输出目录参数，必须提供
@click.option('-d', '--device', default='cuda:0')  # 设备参数，默认为 'cuda:0'，即使用 GPU

def main(checkpoint, device, epochs=50):
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




    with torch.no_grad():
        # 在推理阶段，禁用梯度计算以节省内存和计算资源。

        policy.reset()
        # 重置策略网络（如果需要的话），例如重置内部状态或模型参数。
        import numpy as np

        # 假设你有四张形状为 (128, 128, 3) 的 RGB 图像
        image1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        image3 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        image4 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        # 将这些图像放入一个列表
        images = [image1, image2, image3, image4]

        # 将四张图像堆叠为 (4, 128, 128, 3) 的形状
        images_stack = np.stack(images, axis=0)

        # 现在的形状是 (4, 128, 128, 3)，我们希望将其重塑为 (2, 128, 128, 3) 这种格式
        # 这里 n_obs_steps = 2，意味着我们有 2 张图像分别对应 'wrist' 和 'head'

        # 创建字典并分配
        obs = {
            'wrist': images_stack[:2],  # 选择前两张图像作为 'wrist'
            'head': images_stack[2:],  # 选择后两张图像作为 'head'
        }

        # 查看字典中的数据形状
        print("obs_dict['wrist'].shape:", obs['wrist'].shape)  # 输出: (2, 128, 128, 3)
        print("obs_dict['head'].shape:", obs['head'].shape)  # 输出: (2, 128, 128, 3)

        # 将环境的观察数据（`obs`）转换为模型所需要的输入格式
        obs_dict_np = get_real_obs_dict(
            env_obs=obs, shape_meta=cfg.task.shape_meta)
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

        assert action.shape[-1] == 7




        del result
        # 删除推理结果，以释放内存。



# %%
if __name__ == '__main__':
    main()
