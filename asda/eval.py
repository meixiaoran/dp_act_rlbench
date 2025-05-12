import sys
# 设置标准输出和标准错误的缓冲方式为行缓冲，这样输出会在每行结束时刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click  # 用于命令行参数解析
import hydra  # 用于配置管理和依赖注入
import torch  # 用于加载和操作 PyTorch 模型
import dill  # 用于 pickle 操作（比标准 pickle 支持更多对象类型）
import wandb  # 用于日志和监控
import json  # 用于处理 JSON 数据
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 导入基类，代表工作空间

@click.command()  # 使用 click 库定义一个命令行工具
@click.option('-c', '--checkpoint', required=True)  # 检查点路径参数，必须提供
@click.option('-o', '--output_dir', required=True)  # 输出目录参数，必须提供
@click.option('-d', '--device', default='cuda:0')  # 设备参数，默认为 'cuda:0'，即使用 GPU
def main(checkpoint, output_dir, device):
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
    workspace = cls(cfg, output_dir=output_dir)  # 实例化工作空间对象
    workspace: BaseWorkspace  # 确保 workspace 是 BaseWorkspace 类型
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # 加载模型的 payload 数据到工作空间中
    print(dir(cfg))
    print("name", cfg["name"])
    print("n_action_steps", cfg["n_action_steps"])
    print("n_obs_steps", cfg["n_obs_steps"])
    print("shape_meta", cfg["shape_meta"])
    print("task_name", cfg["task_name"])
    print("obs_as_cond", cfg["obs_as_cond"])

    print("horizon", cfg["horizon"])
    print("exp_name", cfg["exp_name"])
    print("dataset_obs_steps", cfg["dataset_obs_steps"])
    # 获取工作空间中的策略模型
    policy = workspace.model
    if cfg.training.use_ema:  # 如果使用了 EMA（Exponential Moving Average），则使用 EMA 模型
        policy = workspace.ema_model

    # 将模型移到指定的设备（GPU 或 CPU）
    device = torch.device(device)
    policy.to(device)
    policy.eval()  # 设置模型为评估模式（关闭 Dropout 等）


    # # 实例化环境运行器，并在指定的输出目录中运行评估
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,  # 从配置文件中获取环境运行器的配置
        output_dir=output_dir)
    runner_log = env_runner.run(policy)  # 运行评估，获取日志

    # 将日志转换为 JSON 格式
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path  # 如果值是一个视频对象，则存储视频路径
    #     else:
    #         json_log[key] = value  # 否则，直接存储值
    # # 将 JSON 格式的日志写入文件
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

# 如果脚本作为主程序执行，调用 main 函数
if __name__ == '__main__':
    main()
