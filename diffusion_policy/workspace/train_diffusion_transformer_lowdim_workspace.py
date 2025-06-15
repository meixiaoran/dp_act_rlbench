# 主程序入口，通常用于测试或直接运行脚本时调用
from distutils.command.config import config

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    # 获取根目录路径，并将其转换为字符串形式
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)  # 将根目录添加到系统路径中
    os.chdir(ROOT_DIR)  # 切换当前工作目录到根目录

# 导入所需的库
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

# 导入自定义的工具函数和模块
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

# 注册一个新的 OmegaConf 解析器，用于支持 `eval` 函数在配置文件中的使用
OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
# 定义一个用于训练的工作空间类
class TrainDiffusionTransformerLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']  # 在工作空间中包含 global_step 和 epoch 这两个字段

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)  # 调用父类的构造函数

        # 设置随机种子，以保证实验的可复现性
        seed = cfg.training.seed
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        np.random.seed(seed)  # 设置 numpy 的随机种子
        random.seed(seed)  # 设置 Python 内置的 random 模块的随机种子

        # 配置模型
        self.model: DiffusionTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)  # 从配置文件中实例化模型

        # 配置 EMA（指数滑动平均）模型
        self.ema_model: DiffusionTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)  # 如果启用 EMA，深拷贝模型作为 EMA 模型

        # 配置训练优化器
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)  # 从模型中获取优化器

        # 初始化训练状态变量
        self.global_step = 0  # 当前训练步骤
        self.epoch = 0  # 当前训练轮次

    # 运行训练过程
    def run(self):
        # 深拷贝配置文件，避免修改原始配置
        cfg = copy.deepcopy(self.cfg)

        # 如果需要恢复训练，则加载最近的检查点
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()  # 获取最新的检查点路径
            if lastest_ckpt_path.is_file():  # 如果检查点文件存在
                print(f"Resuming from checkpoint {lastest_ckpt_path}")  # 输出恢复信息
                self.load_checkpoint(path=lastest_ckpt_path)  # 从检查点加载模型和状态

        # 配置数据集
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)  # 从配置中实例化数据集
        assert isinstance(dataset, BaseLowdimDataset)  # 确保数据集是期望的类型
        train_dataloader = DataLoader(dataset, **cfg.dataloader)  # 创建训练数据加载器
        normalizer = dataset.get_normalizer()  # 获取数据标准化工具

        # 配置验证数据集
        val_dataset = dataset.get_validation_dataset()  # 获取验证数据集
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)  # 创建验证数据加载器

        # 设置模型的标准化工具
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)  # 如果使用 EMA，也设置 EMA 模型的标准化工具

        # 配置学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,  # 获取学习率调度器配置
            optimizer=self.optimizer,  # 传入优化器
            num_warmup_steps=cfg.training.lr_warmup_steps,  # 获取 warmup 步数
            num_training_steps=(
                                       len(train_dataloader) * cfg.training.num_epochs) \
                               // cfg.training.gradient_accumulate_every,  # 计算总训练步数
            last_epoch=self.global_step - 1  # 设置学习率调度器的最后一次 epoch
        )

        # 配置 EMA 模型
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,  # 从配置中实例化 EMA 模型
                model=self.ema_model  # 传入 EMA 模型
            )

        # 配置环境运行器
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,  # 从配置中实例化环境运行器
            output_dir=self.output_dir  # 设置输出目录
        )
        assert isinstance(env_runner, BaseLowdimRunner)  # 确保环境运行器是期望的类型

        # 配置 WandB 日志记录
        wandb_run = wandb.init(
            dir=str(self.output_dir),  # 设置 WandB 输出目录
            config=OmegaConf.to_container(cfg, resolve=True),  # 将配置文件转化为字典形式并传递给 WandB
            **cfg.logging  # 其他日志相关的配置
        )
        # 更新 WandB 配置中的输出目录信息
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # 配置检查点管理器
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),  # 设置检查点保存目录
            **cfg.checkpoint.topk  # 配置检查点管理的 Top-K 策略
        )

        # 将模型和优化器转移到指定的设备（如 GPU 或 CPU）
        device = torch.device(cfg.training.device)
        self.model.to(device)  # 将模型转移到设备
        if self.ema_model is not None:
            self.ema_model.to(device)  # 如果使用 EMA 模型，将其转移到设备
        optimizer_to(self.optimizer, device)  # 将优化器也转移到设备

        # 保存训练批次（用于采样）
        train_sampling_batch = None

        # 如果是调试模式，修改训练的一些配置参数，减少训练周期
        if cfg.training.debug:
            cfg.training.num_epochs = 2  # 设置训练轮数为 2
            cfg.training.max_train_steps = 3  # 设置最大训练步数为 3
            cfg.training.max_val_steps = 3  # 设置最大验证步数为 3
            cfg.training.rollout_every = 1  # 每 1 步进行一次 rollout
            cfg.training.checkpoint_every = 1  # 每 1 步保存一次检查点
            cfg.training.val_every = 1  # 每 1 步进行一次验证
            cfg.training.sample_every = 1  # 每 1 步进行一次采样

        # 训练循环
        log_path = os.path.join(self.output_dir, 'logs.json.txt')  # 设置日志文件保存路径
        with JsonLogger(log_path) as json_logger:  # 使用 JsonLogger 记录训练日志
            for local_epoch_idx in range(cfg.training.num_epochs):  # 循环训练指定的轮数
                step_log = dict()  # 存储每步的日志信息
                # ========= 本轮训练 ==========
                train_losses = list()  # 存储每步的训练损失
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):  # 遍历训练数据集的每个批次
                        # 数据转移到指定设备（如 GPU 或 CPU）
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch  # 保存一个批次样本用于后续采样

                        # 计算损失
                        raw_loss = self.model.compute_loss(batch)  # 计算模型的原始损失
                        loss = raw_loss / cfg.training.gradient_accumulate_every  # 按照累积梯度步数计算梯度
                        loss.backward()  # 反向传播计算梯度

                        # 更新优化器
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()  # 优化器更新
                            self.optimizer.zero_grad()  # 清空梯度
                            lr_scheduler.step()  # 学习率调度器更新

                        # 更新 EMA 模型（如果启用）
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # 日志记录
                        raw_loss_cpu = raw_loss.item()  # 获取 CPU 上的损失值
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)  # 更新进度条的损失显示
                        train_losses.append(raw_loss_cpu)  # 将当前损失加入到损失列表中
                        step_log = {
                            'train_loss': raw_loss_cpu,  # 当前批次的训练损失
                            'global_step': self.global_step,  # 当前全局训练步数
                            'epoch': self.epoch,  # 当前训练轮次
                            'lr': lr_scheduler.get_last_lr()[0]  # 当前学习率
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))  # 判断是否为当前 epoch 的最后一个批次
                        if not is_last_batch:  # 如果不是最后一个批次
                            # 将日志记录到 WandB 和 JSON 文件
                            wandb_run.log(step_log, step=self.global_step)  # 记录到 WandB
                            json_logger.log(step_log)  # 记录到 JSON 文件
                            self.global_step += 1  # 增加全局步数

                        # 如果达到最大训练步数，提前结束当前 epoch
                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # 每个 epoch 结束后
                # 将训练损失替换为当前 epoch 的平均损失
                train_loss = np.mean(train_losses)  # 计算平均训练损失
                step_log['train_loss'] = train_loss  # 更新日志中的训练损失

                # ========= 当前 epoch 的验证 ==========
                policy = self.model  # 默认使用当前模型进行评估
                if cfg.training.use_ema:  # 如果启用了 EMA 模型，使用 EMA 模型进行评估
                    policy = self.ema_model
                policy.eval()  # 将模型设置为评估模式（关闭 Dropout 和 BatchNorm 等）

                # 进行 Rollout、验证、采样和检查点保存等操作
                if (self.epoch % cfg.training.rollout_every) == 0:  # 每隔指定的轮次进行一次 rollout
                    runner_log = env_runner.run(policy)  # 运行环境进行 rollout
                    # 记录所有的 rollout 日志信息
                    step_log.update(runner_log)

                # 进行验证
                if (self.epoch % cfg.training.val_every) == 0:  # 每隔指定的轮次进行一次验证
                    with torch.no_grad():  # 在验证过程中不计算梯度
                        val_losses = list()  # 存储每个批次的验证损失
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):  # 遍历验证数据集
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))  # 将批次数据转移到设备上
                                loss = self.model.compute_loss(batch)  # 计算当前批次的验证损失
                                val_losses.append(loss)  # 将验证损失加入列表
                                # 如果达到最大验证步数，提前结束验证
                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        # 如果有验证损失，计算平均损失并记录
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()  # 计算平均损失
                            # 记录当前 epoch 的验证损失
                            step_log['val_loss'] = val_loss

                # 在训练批次上进行 Diffusion 采样
                if (self.epoch % cfg.training.sample_every) == 0:  # 每隔指定的轮次进行一次采样
                    with torch.no_grad():  # 不计算梯度
                        # 从训练集采样一个批次，并评估预测结果与真实结果的差异
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = {'obs': batch['obs']}  # 获取观测数据
                        gt_action = batch['action']  # 获取真实动作

                        # 预测动作
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:  # 如果只预测动作步数
                            pred_action = result['action']  # 预测的动作
                            start = cfg.n_obs_steps - 1  # 设置开始的观测步数
                            end = start + cfg.n_action_steps  # 设置结束的动作步数
                            gt_action = gt_action[:, start:end]  # 截取对应的真实动作
                        else:
                            pred_action = result['action_pred']  # 预测的动作序列

                        # 计算预测动作与真实动作之间的均方误差（MSE）
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()  # 记录训练动作的 MSE 错误
                        # 清除不再需要的变量，释放内存
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # 保存检查点
                if (self.epoch % cfg.training.checkpoint_every) == 0:  # 每隔指定的轮次保存一次检查点
                    # 如果需要保存最后的检查点，则保存
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    # 如果需要保存模型快照，则保存
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # 清理度量名称（避免出现不合法的字符）
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')  # 替换掉 / 字符
                        metric_dict[new_key] = value  # 更新日志信息

                    # 获取最优检查点路径
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:  # 如果有最优检查点路径
                        self.save_checkpoint(path=topk_ckpt_path)  # 保存最优检查点

                # 当前 epoch 的评估结束，切换回训练模式
                policy.train()

                # 每个 epoch 结束后
                # 将最后一步的日志与验证和 rollout 日志合并
                wandb_run.log(step_log, step=self.global_step)  # 记录到 WandB
                json_logger.log(step_log)  # 记录到 JSON 文件
                self.global_step += 1  # 增加全局步数
                self.epoch += 1  # 增加当前训练轮次

# Hydra 主函数，配置路径和配置文件
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),  # 配置文件路径
    config_name=pathlib.Path(__file__).stem)  # 配置文件名称与脚本名相同
def main(cfg):
    print(cfg)
    workspace = TrainDiffusionTransformerLowdimWorkspace(cfg)  # 实例化训练工作空间
    workspace.run()  # 运行训练

# 主程序入口
if __name__ == "__main__":
    main()  # 调用主函数启动训练
