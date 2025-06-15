# 导入 sys 模块来修改标准输出流和标准错误流
import sys

# 设置行缓冲模式来处理标准输出和标准错误
# 这里的目的是确保每当有输出时会立即显示，而不是等缓冲区满了才输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# 导入 hydra 配置管理库
import hydra
# 导入 OmegaConf，它是一个配置管理库，用于处理和解析配置文件
from omegaconf import OmegaConf
# 导入 pathlib 用于处理文件路径
import pathlib
# 导入一个名为 BaseWorkspace 的基类，它位于 diffusion_policy.workspace.base_workspace 中
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 注册一个新的 OmegaConf 解析器，使得配置文件中可以使用 ${eval:''} 来执行 Python 代码
OmegaConf.register_new_resolver("eval", eval, replace=True)

# 使用 Hydra 装饰器来将配置文件与 Python 脚本结合
@hydra.main(
    version_base=None,  # 表示 Hydra 版本可以为空，这里会自动推断
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy', 'config'))  # 指定配置文件所在路径
)
def main(cfg: OmegaConf):
    # 解析配置文件，确保所有动态解析的配置（例如 ${now:}）都会使用同一时间
    OmegaConf.resolve(cfg)

    # 根据配置文件中的_target_字段，动态加载对应的类（例如 BaseWorkspace 的子类）
    cls = hydra.utils.get_class(cfg._target_)

    # 实例化该类，并传入解析后的配置文件（cfg）作为参数
    workspace: BaseWorkspace = cls(cfg)

    # 调用实例的 run 方法，执行任务
    workspace.run()

# 当脚本作为主程序执行时，调用 main 函数
if __name__ == "__main__":
    main()
