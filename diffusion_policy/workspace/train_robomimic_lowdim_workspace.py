# 导入必要的库
if __name__ == "__main__":  # 确保只有当该脚本直接运行时，以下代码才会执行
    import sys  # 用于操作Python运行环境和路径
    import os  # 用于操作系统接口，比如文件和目录操作
    import pathlib  # 用于处理路径和文件系统

    # 获取当前脚本的根目录，并将其添加到系统路径中
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)  # 获取当前脚本的根目录（3层上级目录）
    sys.path.append(ROOT_DIR)  # 将根目录添加到Python的模块搜索路径中
    os.chdir(ROOT_DIR)  # 改变当前工作目录为根目录

# 导入其他必要的库
import os  # 文件和目录操作
import hydra  # 配置管理库，支持配置文件的读取和动态解析
import torch  # PyTorch深度学习框架
from omegaconf import OmegaConf  # 配置文件解析库，提供多种高级功能
import pathlib  # 处理路径
from torch.utils.data import DataLoader  # PyTorch的数据加载工具
import copy  # 用于复制对象
import random  # 生成随机数
import wandb  # 用于实验追踪和可视化
import tqdm  # 用于显示进度条
import numpy as np  # 数值计算库
import shutil  # 高级文件操作
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 导入基本工作空间类
from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy  # 导入低维度政策模型
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset  # 导入低维度数据集类
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner  # 导入环境执行类
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager  # 导入检查点管理工具
from diffusion_policy.common.json_logger import JsonLogger  # 导入日志工具
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to  # 导入PyTorch工具

# 注册自定义解析器（这里是eval解析器）
OmegaConf.register_new_resolver("eval", eval, replace=True)

# 定义一个用于训练Robomimic低维度策略的工作空间类
class TrainRobomimicLowdimWorkspace(BaseWorkspace):
    # 该类只包括 'global_step' 和 'epoch' 这两个字段
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        # 调用父类的构造函数，初始化基本工作空间
        super().__init__(cfg)

        # 设置随机种子，确保实验的可重复性
        seed = cfg.training.seed  # 从配置中获取随机种子
        torch.manual_seed(seed)  # 为PyTorch设置随机种子
        np.random.seed(seed)  # 为NumPy设置随机种子
        random.seed(seed)  # 为Python的标准库设置随机种子

        # 配置模型
        self.model: RobomimicLowdimPolicy = hydra.utils.instantiate(cfg.policy)  # 使用hydra来实例化模型

        # 配置训练状态，初始化全局步骤和当前训练周期
        self.global_step = 0  # 初始化全局步骤（用于跟踪训练进度）
        self.epoch = 0  # 初始化当前周期数（epoch）

    def run(self):
        # 创建一个配置的深拷贝，避免直接修改原配置
        cfg = copy.deepcopy(self.cfg)

        # 如果需要恢复训练
        if cfg.training.resume:
            # 获取最新的检查点路径
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():  # 如果检查点文件存在
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)  # 加载最新的检查点继续训练

        # 配置数据集
        dataset: BaseLowdimDataset  # 数据集类型声明
        dataset = hydra.utils.instantiate(cfg.task.dataset)  # 从配置文件中实例化数据集对象
        assert isinstance(dataset, BaseLowdimDataset)  # 确保数据集对象是预期的类型
        train_dataloader = DataLoader(dataset, **cfg.dataloader)  # 配置训练数据加载器
        normalizer = dataset.get_normalizer()  # 获取数据集的标准化器

        # 配置验证数据集
        val_dataset = dataset.get_validation_dataset()  # 获取验证数据集
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)  # 配置验证数据加载器

        # 将标准化器传给模型
        self.model.set_normalizer(normalizer)

        # 配置环境执行器
        env_runner: BaseLowdimRunner  # 环境执行器类型声明
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)  # 从配置文件实例化环境执行器，并指定输出目录
        assert isinstance(env_runner, BaseLowdimRunner)  # 确保环境执行器是预期的类型

        # 配置日志记录
        wandb_run = wandb.init(
            dir=str(self.output_dir),  # 设置输出目录
            config=OmegaConf.to_container(cfg, resolve=True),  # 将配置转换为字典形式传给wandb
            **cfg.logging  # 额外的日志配置
        )
        # 更新wandb的配置，包括输出目录
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # 配置检查点管理器（保存前K个最好的模型）
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),  # 设置检查点保存目录
            **cfg.checkpoint.topk  # 从配置中加载topK检查点参数
        )

        # 设置训练设备（GPU或CPU）
        device = torch.device(cfg.training.device)
        self.model.to(device)  # 将模型转移到指定设备

        # 如果是调试模式，修改训练配置为较小的训练参数
        if cfg.training.debug:
            cfg.training.num_epochs = 2  # 设置训练周期数为2
            cfg.training.max_train_steps = 3  # 设置最大训练步骤为3
            cfg.training.max_val_steps = 3  # 设置最大验证步骤为3
            cfg.training.rollout_every = 1  # 每训练1次进行一次rollout
            cfg.training.checkpoint_every = 1  # 每训练1次保存检查点
            cfg.training.val_every = 1  # 每训练1次进行一次验证

        # 训练循环
        log_path = os.path.join(self.output_dir, 'logs.json.txt')  # 设置日志文件路径
        with JsonLogger(log_path) as json_logger:  # 使用JsonLogger记录训练日志
            for local_epoch_idx in range(cfg.training.num_epochs):  # 遍历每一个epoch
                step_log = dict()  # 每个step的日志
                # ========= 训练一个epoch ==========
                train_losses = list()  # 用于记录每个batch的训练损失
                # 通过tqdm显示训练进度条
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):  # 遍历每个训练batch
                        # 将数据转移到指定设备
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # 模型在当前batch上的训练步骤
                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        # 记录损失并更新进度条
                        loss_cpu = info['losses']['action_loss'].item()  # 获取动作损失
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)  # 更新进度条中的损失显示
                        train_losses.append(loss_cpu)  # 将损失添加到列表中
                        step_log = {
                            'train_loss': loss_cpu,  # 当前训练损失
                            'global_step': self.global_step,  # 当前全局训练步数
                            'epoch': self.epoch  # 当前epoch
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))  # 判断是否是最后一个batch
                        if not is_last_batch:
                            # 如果不是最后一个batch，则记录当前step的日志
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        # 如果达到最大训练步数，提前结束该epoch的训练
                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # 在每个epoch结束时
                # 计算该epoch的平均训练损失
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss  # 更新训练损失为epoch的平均值

                # ========= 评估当前epoch ==========
                self.model.eval()  # 将模型设置为评估模式

                # 每隔一定周期进行rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)  # 运行环境执行器
                    # 更新日志，记录rollout的结果
                    step_log.update(runner_log)

                # 每隔一定周期进行验证
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():  # 在验证时不计算梯度
                        val_losses = list()  # 用于记录每个验证batch的损失
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):  # 遍历验证数据集的batch
                                # 将数据转移到指定设备
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # 在验证模式下进行训练（这里的train_on_batch实际上是验证）
                                info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                loss = info['losses']['action_loss']  # 获取验证损失
                                val_losses.append(loss)  # 记录损失
                                # 如果达到最大验证步数，提前结束验证
                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        # 计算并记录验证损失的平均值
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss  # 更新验证损失为epoch的平均值

                # 如果满足配置中的保存检查点条件
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # 保存检查点
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()  # 保存最新的检查点
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()  # 保存当前模型的快照

                    # 清理指标名称
                    metric_dict = dict()  # 创建一个新的字典用于存储指标
                    for key, value in step_log.items():  # 遍历当前step的日志
                        new_key = key.replace('/', '_')  # 将指标名称中的斜杠替换为下划线
                        metric_dict[new_key] = value  # 更新字典，避免名称冲突

                    # 此处无法复制最后的检查点文件
                    # 原因是save_checkpoint函数是使用线程实现的，此时文件可能尚未完全写入！
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)  # 获取符合条件的TopK检查点路径

                    # 如果返回了一个检查点路径，保存模型到该路径
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # ========= 结束本epoch的评估 ==========
                self.model.train()  # 切换模型到训练模式（与评估模式相对）

                # epoch结束操作
                # 将当前step的日志，包括验证和rollout结果，记录到wandb和JsonLogger
                wandb_run.log(step_log, step=self.global_step)  # 使用wandb记录日志
                json_logger.log(step_log)  # 使用JsonLogger记录日志
                self.global_step += 1  # 更新全局训练步数
                self.epoch += 1  # 更新当前训练周期数


# 使用Hydra框架的装饰器来定义main函数
@hydra.main(
    version_base=None,  # 指定Hydra配置系统的版本（None表示不指定版本）
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),  # 配置文件所在路径，当前脚本的上两级目录下的"config"目录
    config_name=pathlib.Path(__file__).stem)  # 配置文件的名称，使用当前脚本的文件名（不带扩展名）作为配置文件名
def main(cfg):
    # 实例化训练工作空间
    workspace = TrainRobomimicLowdimWorkspace(cfg)
    workspace.run()  # 执行训练过程

# 当脚本作为主程序运行时，调用main函数
if __name__ == "__main__":
    main()  # 启动主函数

