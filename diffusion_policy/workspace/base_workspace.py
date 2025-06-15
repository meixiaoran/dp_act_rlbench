from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest',
                        exclude_keys=None,
                        include_keys=None,
                        use_thread=True):
        # 如果没有提供保存路径，则使用默认路径，路径为 'output_dir/checkpoints/tag.ckpt'
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)  # 如果提供了路径，转换为 Path 对象

        # 如果没有提供 exclude_keys，则使用默认的 exclude_keys
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)

        # 如果没有提供 include_keys，则默认包含 '_output_dir'
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        # 确保保存路径的父目录存在，如果不存在则创建
        path.parent.mkdir(parents=False, exist_ok=True)

        # 创建一个字典 payload 用于保存配置和状态字典等信息
        payload = {
            'cfg': self.cfg,  # 保存配置文件（cfg）
            'state_dicts': dict(),  # 用于存储模型、优化器、采样器等的状态字典
            'pickles': dict()  # 用于存储其他需要序列化的对象（pickle序列化）
        }

        # 遍历当前对象的属性字典
        for key, value in self.__dict__.items():
            # 如果属性有 'state_dict' 和 'load_state_dict' 方法（通常是模型、优化器等）
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # 如果该属性的 key 不在排除列表中，保存其状态字典
                if key not in exclude_keys:
                    if use_thread:  # 如果使用线程进行保存
                        # 将状态字典保存到 payload 中，并将其复制到 CPU（如果在 GPU 上）
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        # 不使用线程，则直接保存状态字典
                        payload['state_dicts'][key] = value.state_dict()
            # 如果属性的 key 在 include_keys 中，则需要序列化并保存
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)  # 使用 dill 序列化该属性

        # 如果使用线程保存
        if use_thread:
            # 创建一个新的线程来执行保存操作，这样可以避免主线程被阻塞
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill)
            )
            self._saving_thread.start()  # 启动线程
        else:
            # 如果不使用线程，则直接执行保存操作
            torch.save(payload, path.open('wb'), pickle_module=dill)

        # 返回保存文件的绝对路径
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
