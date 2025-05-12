if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

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
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']  # 定义需要包含的键，用于跟踪训练过程中的全局步骤（global_step）和当前训练轮次（epoch）

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)  # 调用父类的初始化方法，传入配置文件和输出目录

        # 设置随机种子，确保实验的可复现性
        seed = cfg.training.seed  # 从配置文件中获取随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        random.seed(seed)  # 设置 Python 内建的 random 模块的随机种子

        # 配置模型
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)  # 使用 Hydra 实例化模型（根据配置文件）

        # 配置指数移动平均（EMA）模型
        self.ema_model: DiffusionTransformerHybridImagePolicy = None  # 初始化 ema_model 为 None
        if cfg.training.use_ema:  # 如果配置中启用了 EMA
            self.ema_model = copy.deepcopy(self.model)  # 通过深拷贝将模型的参数复制到 ema_model

        # 配置优化器
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)  # 获取优化器，通常是基于配置中的优化器类型和参数

        # 配置训练状态（用于跟踪训练进度）
        self.global_step = 0  # 初始化全局步骤（每处理一个批次时增加）
        self.epoch = 0  # 初始化当前训练轮次（epoch）

    def run(self):
        cfg = copy.deepcopy(self.cfg)  # 复制配置文件，以防在训练过程中修改原始配置

        # 如果需要恢复训练
        if cfg.training.resume:  # 如果配置中启用了恢复训练
            lastest_ckpt_path = self.get_checkpoint_path()  # 获取最新的检查点路径
            if lastest_ckpt_path.is_file():  # 如果检查点文件存在
                print(f"Resuming from checkpoint {lastest_ckpt_path}")  # 打印恢复训练的提示
                self.load_checkpoint(path=lastest_ckpt_path)  # 从最新检查点加载模型

        # 配置数据集
        dataset: BaseImageDataset  # 声明数据集类型为 BaseImageDataset
        print(cfg.task.dataset)
        dataset = hydra.utils.instantiate(cfg.task.dataset)  # 使用 Hydra 实例化数据集（根据配置文件）

        assert isinstance(dataset, BaseImageDataset)  # 确保数据集是 BaseImageDataset 类型
        train_dataloader = DataLoader(dataset, **cfg.dataloader)  # 配置训练数据加载器
        normalizer = dataset.get_normalizer()  # 获取数据集的归一化器（用于对图像数据进行标准化）

        # 配置验证数据集
        val_dataset = dataset.get_validation_dataset()  # 获取验证集
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)  # 配置验证数据加载器

        # 设置模型的归一化器
        self.model.set_normalizer(normalizer)  # 将归一化器应用于训练模型
        if cfg.training.use_ema:  # 如果启用了 EMA
            self.ema_model.set_normalizer(normalizer)  # 也将归一化器应用于 EMA 模型

        # 配置学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,  # 配置文件中指定的学习率调度器
            optimizer=self.optimizer,  # 传入优化器
            num_warmup_steps=cfg.training.lr_warmup_steps,  # 预热步数
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,  # 计算总的训练步骤数
            last_epoch=self.global_step-1  # 设置调度器的起始步数
        )

        # 配置 EMA 模型
        ema: EMAModel = None  # 初始化 EMA 模型为 None
        if cfg.training.use_ema:  # 如果启用了 EMA
            ema = hydra.utils.instantiate(  # 实例化 EMA 模型
                cfg.ema,
                model=self.ema_model)  # 使用 ema_model 来初始化 EMA 模型

        # 配置环境运行器
        # env_runner: BaseImageRunner  # 声明环境运行器类型为 BaseImageRunner
        # env_runner = hydra.utils.instantiate(  # 使用 Hydra 实例化环境运行器
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)  # 传入输出目录
        # assert isinstance(env_runner, BaseImageRunner)  # 确保环境运行器是 BaseImageRunner 类型

        # 配置日志记录
        wandb_run = wandb.init(
            dir=str(self.output_dir),  # 设置输出目录
            config=OmegaConf.to_container(cfg, resolve=True),  # 将配置转换为字典格式
            **cfg.logging  # 配置日志参数
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,  # 更新 WandB 配置，记录输出目录
            }
        )

        # 配置检查点管理
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),  # 保存检查点的目录
            **cfg.checkpoint.topk  # 配置 top-k 检查点参数
        )

        # 设置设备（GPU 或 CPU）
        device = torch.device(cfg.training.device)  # 获取训练设备（比如 GPU 或 CPU）
        self.model.to(device)  # 将模型转移到指定设备
        if self.ema_model is not None:  # 如果有 EMA 模型
            self.ema_model.to(device)  # 也将 EMA 模型转移到指定设备
        optimizer_to(self.optimizer, device)  # 将优化器转移到指定设备

        # 用于采样的训练批次数据
        train_sampling_batch = None
        print("bbbbbb")
        # 如果进入调试模式，设置调试参数
        if cfg.training.debug:
            cfg.training.num_epochs = 2  # 设置训练轮次为 2
            cfg.training.max_train_steps = 3  # 设置最大训练步数为 3
            cfg.training.max_val_steps = 3  # 设置最大验证步数为 3
            cfg.training.rollout_every = 1  # 每隔 1 步进行一次采样
            cfg.training.checkpoint_every = 1  # 每隔 1 步保存一次检查点
            cfg.training.val_every = 1  # 每隔 1 步进行一次验证
            cfg.training.sample_every = 1  # 每隔 1 步进行一次采样

        # 训练循环
        log_path = os.path.join(self.output_dir, 'logs.json.txt')  # 定义日志文件的路径，保存在输出目录下
        with JsonLogger(log_path) as json_logger:  # 使用 JsonLogger 记录训练日志
            # 遍历所有训练轮次（epoch）
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()  # 定义一个字典来存储每个训练步骤的日志信息

                # ========= 本轮次训练 ==========
                train_losses = list()  # 用于存储每个批次的训练损失
                # 使用 tqdm 来显示训练进度条，并控制进度条的刷新频率
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    # 遍历训练数据的每个批次
                    for batch_idx, batch in enumerate(tepoch):



                        # 将当前批次的数据转移到指定设备（CPU 或 GPU）
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        if train_sampling_batch is None:
                            train_sampling_batch = batch  # 保存第一批数据用于后续采样

                        # 计算损失
                        raw_loss = self.model.compute_loss(batch)  # 调用模型的 compute_loss 方法计算当前批次的损失
                        loss = raw_loss / cfg.training.gradient_accumulate_every  # 如果使用梯度累积，进行梯度平均
                        loss.backward()  # 反向传播，计算梯度

                        # 更新优化器

                        self.optimizer.step()  # 更新优化器
                        self.optimizer.zero_grad()  # 清空梯度
                        lr_scheduler.step()  # 更新学习率调度器


                        # 更新 EMA 模型
                        if cfg.training.use_ema:
                            ema.step(self.model)  # 更新 EMA 模型的参数

                        # 记录日志
                        raw_loss_cpu = raw_loss.item()  # 获取 CPU 上的损失值
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)  # 更新进度条显示的损失
                        train_losses.append(raw_loss_cpu)  # 将当前批次的损失添加到训练损失列表中

                        # 构建当前步骤的日志信息
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]  # 记录当前的学习率
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))  # 判断是否为当前 epoch 的最后一个批次
                        if not is_last_batch:
                            # 对于不是最后一个批次的情况，记录日志
                            wandb_run.log(step_log, step=self.global_step)  # 使用 WandB 记录日志
                            json_logger.log(step_log)  # 使用 JsonLogger 记录日志
                            self.global_step += 1  # 更新全局步骤数

                        # 如果达到最大训练步数，则停止当前 epoch
                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                                cfg.training.max_train_steps - 1):
                            break

                # 计算当前 epoch 的平均训练损失
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss  # 更新日志字典中的训练损失为平均损失

                # ========= 本轮次评估 ==========
                policy = self.model  # 默认使用当前训练的模型进行评估
                if cfg.training.use_ema:  # 如果启用了 EMA，则使用 EMA 模型进行评估
                    policy = self.ema_model
                policy.eval()  # 切换到评估模式，禁用 Dropout 等操作


                # 如果当前 epoch 需要进行 rollout（生成样本）
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)  # 在环境中运行模型进行采样
                #     # 更新当前步骤的日志信息

                # step_log.update(runner_log)

                # 如果当前 epoch 需要进行验证
                # if (self.epoch % cfg.training.val_every) == 0:
                #     with torch.no_grad():  # 在验证过程中禁用梯度计算，以节省内存
                #         val_losses = list()  # 用于存储验证损失
                #         # 使用 tqdm 显示验证过程的进度条
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                #                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))  # 将验证数据转移到设备上
                #                 loss = self.model.compute_loss(batch)  # 计算验证损失
                #                 val_losses.append(loss)  # 添加到验证损失列表中
                #                 if (cfg.training.max_val_steps is not None) and batch_idx >= (
                #                         cfg.training.max_val_steps - 1):
                #                     break
                #         if len(val_losses) > 0:  # 如果存在验证损失
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()  # 计算验证损失的平均值
                #             # 更新当前步骤的日志信息，记录验证损失
                #             step_log['val_loss'] = val_loss
                #
                # 如果当前 epoch 需要进行采样（例如，从训练集预测动作并计算误差）
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():  # 在采样时禁用梯度计算
                #         # 使用保存的训练批次数据进行采样
                #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                #         obs_dict = batch['obs']  # 获取观察数据
                #         gt_action = batch['action']  # 获取真实的动作
                #
                #         # 使用模型进行预测
                #         result = policy.predict_action(obs_dict)
                #         pred_action = result['action_pred']  # 预测的动作
                #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)  # 计算均方误差（MSE）
                #         step_log['train_action_mse_error'] = mse.item()  # 记录训练集上的动作预测误差
                #         del batch  # 删除临时变量以释放内存
                #         del obs_dict
                #         del gt_action
                #         del result
                #         del pred_action
                #         del mse

                # ========= 保存检查点 ==========
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # 检查点保存
                    if cfg.checkpoint.save_last_ckpt:  # 如果配置中要求保存最后的检查点
                        self.save_checkpoint()  # 保存当前模型的检查点
                        tag = f"{self.epoch}"
                        self.save_checkpoint(tag=tag)
                    if cfg.checkpoint.save_last_snapshot:  # 如果配置中要求保存最后的快照
                        self.save_snapshot()  # 保存当前模型的快照

                    # # 对日志中的度量值名称进行处理，确保没有非法字符（如斜杠）
                    # metric_dict = dict()
                    # for key, value in step_log.items():
                    #     new_key = key.replace('/', '_')  # 将斜杠替换为下划线
                    #     metric_dict[new_key] = value
                    #
                    # # 获取 top-k 检查点路径并保存
                    # print("metric_dict", metric_dict)
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    # if topk_ckpt_path is not None:  # 如果返回了一个有效的 top-k 检查点路径
                    #     self.save_checkpoint(path=topk_ckpt_path)  # 保存 top-k 检查点

                # ========= 结束当前 epoch ==========
                policy.train()  # 切换回训练模式

                # 记录当前 epoch 结束时的日志信息
                wandb_run.log(step_log, step=self.global_step)  # 使用 WandB 记录日志
                json_logger.log(step_log)  # 使用 JsonLogger 记录日志
                self.global_step += 1  # 增加全局步骤数
                self.epoch += 1  # 增加当前训练轮次（epoch）


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()