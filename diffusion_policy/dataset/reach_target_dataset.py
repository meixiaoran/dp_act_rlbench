# 引入必要的模块和库
from typing import Dict, List  # 导入类型注解，方便在函数中定义输入输出的类型。
import torch  # 导入 PyTorch，用于深度学习的张量操作和神经网络模型的构建。
import numpy as np  # 导入 NumPy，用于高效的数组操作和数学计算。
import h5py  # 导入 h5py，用于读写 HDF5 文件格式的数据。
from tqdm import tqdm  # 导入 tqdm，用于显示进度条，便于跟踪长时间运行的任务。
import zarr  # 导入 zarr，用于高效的存储和读取大数据，尤其是用于数组的存储。
import os  # 导入 os 模块，提供与操作系统交互的功能，如文件操作等。
import shutil  # 导入 shutil，提供高层次的文件和目录操作，如复制文件、删除文件等。
import copy  # 导入 copy，用于复制对象，包括浅拷贝和深拷贝。
import json  # 导入 json，用于处理 JSON 格式的数据，包括解析和生成 JSON。
import hashlib  # 导入 hashlib，用于进行哈希算法操作，比如计算文件的哈希值。
from filelock import FileLock  # 导入 FileLock，用于在多线程或多进程环境中实现文件级别的锁。
from threadpoolctl import threadpool_limits  # 导入 threadpool_limits，用于控制线程池中线程的数量。
import concurrent.futures  # 导入 concurrent.futures，提供高层次的异步任务执行接口，用于多线程或多进程任务。
import multiprocessing  # 导入 multiprocessing，用于并行计算，通过多进程实现任务并行化。
from omegaconf import OmegaConf  # 导入 OmegaConf，用于配置管理，通常用于加载和处理配置文件（如 YAML 格式）。
from diffusion_policy.common.pytorch_util import dict_apply  # 从 diffusion_policy 中导入自定义的字典处理工具（可能用于递归地应用某些操作到字典）。
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer  # 导入自定义的数据集类和归一化类。
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # 导入归一化类。
from diffusion_policy.model.common.rotation_transformer import RotationTransformer  # 导入旋转变换类，用于图像处理或其他几何变换。
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k  # 导入图像编解码器相关函数和类，用于图像数据的编码解码。
from diffusion_policy.common.replay_buffer import ReplayBuffer  # 导入重放缓冲区类，用于存储和管理经验数据。
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask  # 导入采样器和辅助函数，用于生成训练数据的采样顺序。
from diffusion_policy.common.normalize_util import (  # 导入归一化工具函数，用于处理不同类型的标准化。
    robomimic_abs_action_only_normalizer_from_stat,  # 用于标准化机器人动作数据。
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,  # 用于双臂机器人的动作标准化。
    get_range_normalizer_from_stat,  # 用于获取范围归一化器。
    get_image_range_normalizer,  # 用于图像的范围归一化。
    get_identity_normalizer_from_stat,  # 用于标识符的归一化处理。
    array_to_stats  # 将数组转换为统计信息的函数，可能用于计算均值、方差等。
)

# 注册自定义图像编解码器
register_codecs()  # 调用注册函数，将 Jpeg2k 或其他图像编码器加入系统，以便后续使用。



# 定义 RobomimicReplayImageDataset 类，继承自 BaseImageDataset 类
class reach_target_dataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,  # 存储数据形状和其他元信息的字典
                 dataset_path: str,  # 数据集的文件路径
                 horizon=1,  # 时间步长，默认值为1
                 pad_before=0,  # 在每个序列开始前填充的步数
                 pad_after=0,  # 在每个序列结束后填充的步数
                 n_obs_steps=None,  # 观察步骤数，可能影响每次使用的数据长度
                 abs_action=False,  # 是否使用绝对动作（如果为 True，则使用绝对动作）
                 rotation_rep='rotation_6d',  # 旋转表示法，默认为 'rotation_6d'，当 abs_action=False 时该参数被忽略
                 use_legacy_normalizer=False,  # 是否使用旧的归一化方法
                 use_cache=False,  # 是否使用缓存
                 seed=42,  # 随机种子，用于数据的随机性
                 val_ratio=0.0  # 验证集比例，默认为 0，表示没有验证集
                 ):

        # 初始化旋转变换器，使用 'axis_angle' 到 'rotation_6d' 的旋转表示转换
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None  # 初始化回放缓冲区为空
        if use_cache:  # 如果启用了缓存
            # 定义缓存路径和锁文件路径
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')  # 输出缓存锁定信息

            # 使用文件锁来确保在多进程环境下只有一个进程能够创建或读取缓存
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):  # 如果缓存文件不存在
                    # 缓存不存在，进行创建
                    try:
                        print('Cache does not exist. Creating!')  # 输出缓存不存在的提示信息
                        # 这里注释掉的代码本来是用于创建 Zarr 存储的，但没有启用
                        # store = zarr.DirectoryStore(cache_zarr_path)

                        # 调用 _convert_robomimic_to_replay 函数，将 Robomimic 数据集转换为 ReplayBuffer
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),  # 使用内存存储来临时保存数据
                            shape_meta=shape_meta,  # 传入数据形状元信息
                            dataset_path=dataset_path,  # 数据集路径
                            abs_action=abs_action,  # 是否使用绝对动作
                            rotation_transformer=rotation_transformer  # 使用旋转变换器
                        )
                        print('Saving cache to disk.')  # 输出保存缓存到磁盘的提示信息

                        # 将生成的 replay_buffer 保存到 Zip 格式的 Zarr 存储
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store  # 保存数据到 ZipStore
                            )
                    except Exception as e:  # 如果在创建缓存过程中出现错误，删除缓存并重新抛出异常
                        shutil.rmtree(cache_zarr_path)  # 删除缓存文件夹
                        raise e  # 抛出异常
                else:
                    # 如果缓存文件已经存在，直接加载缓存
                    print('Loading cached ReplayBuffer from Disk.')  # 输出加载缓存的提示信息

                    # 从缓存的 ZipStore 加载回放缓冲区数据
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store,  # 从存储中复制数据
                            store=zarr.MemoryStore()  # 使用内存存储
                        )
                    print('Loaded!')  # 输出缓存加载完成的提示信息


        else:

            # 如果缓存存在，直接加载缓存的 ReplayBuffer

            # 如果没有使用缓存，则会创建新的 ReplayBuffer

            replay_buffer = _convert_robomimic_to_replay(

                store=zarr.MemoryStore(),  # 使用内存存储来临时保存数据

                shape_meta=shape_meta,  # 数据形状元信息

                dataset_path=dataset_path,  # 数据集的路径

                abs_action=abs_action,  # 是否使用绝对动作

                rotation_transformer=rotation_transformer  # 旋转变换器

            )

        # 初始化 rgb_keys 和 lowdim_keys 两个列表，分别存储图像数据和低维度数据的键

        rgb_keys = list()  # 用于存储所有 rgb 类型的数据键

        lowdim_keys = list()  # 用于存储所有低维度数据的键

        obs_shape_meta = shape_meta['obs']  # 获取观测数据的形状元信息

        # 遍历观测数据的形状元信息，按类型分配数据键

        for key, attr in obs_shape_meta.items():

            type = attr.get('type', 'low_dim')  # 获取数据类型，默认为 'low_dim'

            if type == 'rgb':  # 如果是 rgb 类型的图像数据

                rgb_keys.append(key)  # 将键添加到 rgb_keys 列表

            elif type == 'low_dim':  # 如果是低维度的数据

                lowdim_keys.append(key)  # 将键添加到 lowdim_keys 列表

        # key_first_k 字典用于存储每个键对应的第一步观测数

        key_first_k = dict()

        if n_obs_steps is not None:  # 如果指定了 n_obs_steps（每个图像数据取前 k 步）

            # 只取图像数据的前 n_obs_steps 步

            for key in rgb_keys + lowdim_keys:  # 遍历所有的 rgb 数据和低维数据

                key_first_k[key] = n_obs_steps  # 将每个键对应的观测步数设置为 n_obs_steps

        # 生成验证集掩码 (val_mask)，用于区分训练集和验证集

        val_mask = get_val_mask(

            n_episodes=replay_buffer.n_episodes,  # 总的实验次数

            val_ratio=val_ratio,  # 验证集的比例

            seed=seed  # 随机种子，用于保证可重复性

        )

        train_mask = ~val_mask  # 训练集掩码是验证集掩码的反向，表示训练集中的数据

        # 创建 SequenceSampler 采样器，用于从回放缓冲区中按顺序采样数据

        sampler = SequenceSampler(

            replay_buffer=replay_buffer,  # 使用的回放缓冲区

            sequence_length=horizon,  # 每次采样的序列长度（一个 episode 的时间步长）

            pad_before=pad_before,  # 在序列前面填充的步数

            pad_after=pad_after,  # 在序列后面填充的步数

            episode_mask=train_mask,  # 训练集的掩码，控制哪些 episode 用于训练

            key_first_k=key_first_k  # 针对每个数据键，只采样前 n_obs_steps 步

        )

        # 将所有配置和参数保存到类的实例中

        self.replay_buffer = replay_buffer  # 保存回放缓冲区

        self.sampler = sampler  # 保存数据采样器

        self.shape_meta = shape_meta  # 保存数据形状元信息

        self.rgb_keys = rgb_keys  # 保存所有 rgb 数据的键

        self.lowdim_keys = lowdim_keys  # 保存所有低维度数据的键

        self.abs_action = abs_action  # 是否使用绝对动作

        self.n_obs_steps = n_obs_steps  # 每次观测的步数

        self.train_mask = train_mask  # 训练集的掩码

        self.horizon = horizon  # 每个序列的时间步长

        self.pad_before = pad_before  # 填充前面步数

        self.pad_after = pad_after  # 填充后面步数

        self.use_legacy_normalizer = use_legacy_normalizer  # 是否使用旧版的归一化器

    def get_validation_dataset(self):
        # 创建一个新的实例 val_set，作为当前数据集（self）的副本
        val_set = copy.copy(self)  # 使用 copy.copy 创建副本，避免直接修改原数据集
        # 设置新的采样器，只采样验证集数据（episode_mask 为训练集掩码的反向）
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,  # 使用原数据集中的回放缓冲区
            sequence_length=self.horizon,  # 序列的长度，保持与训练集相同
            pad_before=self.pad_before,  # 填充前面的步数
            pad_after=self.pad_after,  # 填充后面的步数
            episode_mask=~self.train_mask  # 只使用验证集的样本（train_mask 的反向）
        )
        # 更新验证集的 train_mask 为训练集掩码的反向，确保验证集掩码与训练集掩码不同
        val_set.train_mask = ~self.train_mask
        return val_set  # 返回新的验证数据集对象

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()  # 初始化一个线性归一化器对象

        # 处理动作数据的归一化
        stat = array_to_stats(self.replay_buffer['action'])  # 获取动作数据的统计信息（如均值和标准差）

        if self.abs_action:  # 如果使用绝对动作表示
            if stat['mean'].shape[-1] > 10:  # 如果均值的最后一个维度大于10，认为是双臂动作
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:  # 否则认为是单臂动作
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:  # 如果使用旧的归一化器
                this_normalizer = normalizer_from_stat(stat)  # 使用旧的归一化方法

        else:  # 如果没有使用绝对动作，动作数据已经归一化
            this_normalizer = get_range_normalizer_from_stat(stat)  # 使用恒等归一化器，表示不做额外处理

        normalizer['action'] = this_normalizer  # 将动作数据的归一化器添加到归一化器字典中

        # 处理观测数据的归一化
        for key in self.lowdim_keys:  # 遍历所有低维度数据的键（如位置、速度等）
            stat = array_to_stats(self.replay_buffer[key])  # 获取每个数据键的统计信息

            if key.endswith('pos'):  # 如果是位置数据（通常是关节位置）
                this_normalizer = get_range_normalizer_from_stat(stat)  # 使用范围归一化器
            elif key.endswith('quat'):  # 如果是四元数数据（通常在 [-1, 1] 范围内）
                this_normalizer = get_identity_normalizer_from_stat(stat)  # 使用恒等归一化器
            elif key.endswith('qpos'):  # 如果是关节位置数据（通常是 qpos）
                this_normalizer = get_range_normalizer_from_stat(stat)  # 使用范围归一化器
            else:
                raise RuntimeError('unsupported')  # 如果数据类型不支持，抛出异常
            normalizer[key] = this_normalizer  # 将每个观测数据的归一化器添加到归一化器字典中

        # 处理图像数据的归一化
        for key in self.rgb_keys:  # 遍历所有 RGB 图像数据的键
            normalizer[key] = get_image_range_normalizer()  # 使用图像范围归一化器

        return normalizer  # 返回构造好的归一化器

    def get_all_actions(self) -> torch.Tensor:
        # 返回整个回放缓冲区中的所有动作数据，并转换为 torch.Tensor 格式
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        # 返回数据集的大小，即采样器的长度
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取指定索引 idx 处的数据项

        threadpool_limits(1)  # 限制线程池中并行线程的数量为 1，防止并发操作影响性能
        # 从采样器中获取一个序列数据，idx 表示当前要获取的样本的索引
        data = self.sampler.sample_sequence(idx)

        # 为了节省内存，只返回前 n_obs_steps 个观测数据
        # 如果 self.n_obs_steps 为 None，则此切片操作不做任何处理（即返回全部数据）
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()  # 存储所有观测数据的字典
        for key in self.rgb_keys:  # 遍历所有 RGB 图像数据的键
            # 将图像数据的通道从最后一维移到第一维
            # 例如原始格式为 T,H,W,C，转换为 T,C,H,W
            # 同时将 uint8 类型的图像数据转换为 float32 类型，并归一化到 [0, 1] 范围
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            # 删除原始数据中的图像数据，以节省内存
            del data[key]

        for key in self.lowdim_keys:  # 遍历所有低维数据的键（例如位置、速度等）
            # 获取低维观测数据并转换为 float32 类型
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # 删除原始数据中的低维数据，以节省内存
            del data[key]

        # 构建并返回一个字典，包含了当前观测数据和对应的动作数据
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),  # 将观测数据转换为 torch.Tensor 格式
            'action': torch.from_numpy(data['action'].astype(np.float32))  # 将动作数据转换为 torch.Tensor 格式
        }
        return torch_data

def _convert_actions(raw_actions, abs_action, rotation_transformer):
    # 将原始动作数据转换为适合模型的格式
    actions = raw_actions
    if abs_action:  # 如果使用绝对动作表示
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:  # 如果动作维度为 14，表示双臂任务
            # 将动作数据 reshape 为每个任务包含两个七维动作
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        # 将动作数据分为位置、旋转和夹爪部分
        pos = raw_actions[..., :3]  # 提取位置部分
        rot = raw_actions[..., 3:6]  # 提取旋转部分
        gripper = raw_actions[..., 6:]  # 提取夹爪部分
        # 使用旋转变换器将旋转部分转换为所需的表示
        rot = rotation_transformer.forward(rot)
        # 重新组合转换后的动作数据
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        # if is_dual_arm:  # 如果是双臂任务，将动作 reshape 为每个任务 20 维
        #     raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions  # 返回处理后的动作数据
    return actions

def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
                                 n_workers=None, max_inflight_tasks=None):
    # 将 Robomimic 数据集转换为 ReplayBuffer 格式的数据，并存储在指定的 Zarr 存储中
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()  # 使用的工作线程数默认为 CPU 核心数
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5  # 最大任务数设置为线程数的五倍

    # 解析 shape_meta 中关于观测数据和动作数据的元信息
    rgb_keys = list()  # 存储 RGB 图像数据的键
    lowdim_keys = list()  # 存储低维数据的键（如位置、速度等）
    # 构造压缩器和切片信息
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']  # 获取该数据的形状
        type = attr.get('type', 'low_dim')  # 获取数据类型，默认为 'low_dim'
        if type == 'rgb':  # 如果数据类型为 RGB 图像
            rgb_keys.append(key)
        elif type == 'low_dim':  # 如果数据类型为低维数据
            lowdim_keys.append(key)
    # 创建 Zarr 存储根对象
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)  # 存储数据的组
    meta_group = root.require_group('meta', overwrite=True)  # 存储元数据的组

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']  # 打开 HDF5 文件并读取数据
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']  # 每个 demo_{i} 表示一个轨迹或示范数据
            episode_length = demo['action'].shape[0]  # 计算每个 demo（一个轨迹或示范数据）的长度，
            episode_end = prev_end + episode_length  # 将其结束位置存储在 episode_ends 中
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]  # 取episode_ends中的最后一个元素
        episode_starts = [0] + episode_ends[:-1]  # episode_ends[:-1]表示从第一个取到倒数第二个元素
        _ = meta_group.array('episode_ends', episode_ends,
                             dtype=np.int64, compressor=None,
                             overwrite=True)  # 通过 episode_ends 计算总步骤数 n_steps，并将轨迹的结束位置存储在 meta_group 中
        # save lowdim data  加载低维数据（动作和低维观测）
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            # 对于每个低维数据键（robot0_eef_pos, robot0_gripper_qpos 等）以及动作（action），从每个 demo 中读取数据并将其合并到一个大的数组 this_data 中
            # 每种数据分别放在一个 this_data 中
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'action'
            this_data = list()
            for i in range(len(demos)):  # 遍历每个示范（demo_{i}）
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))  # 读取全部的动作或低维观测数据
            this_data = np.concatenate(this_data, axis=0)  # 将 this_data 列表中的多个数组沿着 axis=0（即按时间步）进行拼接。
            print("afsagsgfghdfhs", len(this_data))

            if key == 'action':  # 如果是动作数据，转换动作数据格式
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                # 确保观测数据的形状符合要求
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])


            # 将数据保存到 Zarr 存储中
            _ = data_group.array(
                name=key,  # 数据名称
                data=this_data,  # 数据内容
                shape=this_data.shape,  # 数据形状
                chunks=this_data.shape,  # 切片信息
                compressor=None,  # 压缩算法
                dtype=this_data.dtype  # 数据类型
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            # 该函数负责将 HDF5 文件中的图像数据复制到 Zarr 存储中
            try:
                # 将 HDF5 数组中的数据复制到 Zarr 数组
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # 确保数据可以成功解码，如果无法解码则会抛出异常
                _ = zarr_arr[zarr_idx]
                return True  # 如果复制成功，返回 True
            except Exception as e:
                # 如果发生异常，返回 False
                return False

        # 使用 tqdm 来显示进度条，表示正在加载图像数据
        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # 每个线程处理一个切片，因此不需要同步
            # 使用 ThreadPoolExecutor 来并行处理图像数据加载，最多允许 n_workers 个线程并行
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()  # 用来保存异步任务的集合
                for key in rgb_keys:  # 遍历所有 RGB 图像数据的键
                    data_key = 'obs/' + key  # 获取图像数据的键（obs/ + key）
                    shape = tuple(shape_meta['obs'][key]['shape'])  # 获取图像的形状
                    c, h, w = shape  # 将图像形状拆分为 c (通道数), h (高度), w (宽度)
                    this_compressor = Jpeg2k(level=50)  # 使用 Jpeg2k 压缩算法进行压缩，设置压缩级别为 50
                    # 创建一个 Zarr 数据集来存储图像数据
                    img_arr = data_group.require_dataset(
                        name=key,  # 数据集名称
                        shape=(n_steps, h, w, c),  # 数据集的总形状 (n_steps, 高度, 宽度, 通道数)
                        chunks=(1, h, w, c),  # 设置切片大小
                        compressor=this_compressor,  # 设置压缩器
                        dtype=np.uint8  # 数据类型为 uint8
                    )

                    # 遍历每一个演示数据
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']  # 获取当前演示的数据
                        hdf5_arr = demo['obs'][key]  # 获取当前演示的图像数据
                        for hdf5_idx in range(hdf5_arr.shape[0]):  # 遍历当前演示中的每一帧图像
                            if len(futures) >= max_inflight_tasks:  # 如果当前的异步任务数超过最大并发数，则等待任务完成
                                # 等待至少一个任务完成，确保不会有太多任务并发执行
                                completed, futures = concurrent.futures.wait(futures,
                                                                             return_when=concurrent.futures.FIRST_COMPLETED)
                                # 检查已完成任务的结果，如果复制失败则抛出异常
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))  # 更新进度条

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx  # 计算对应的 Zarr 索引位置
                            # 提交一个异步任务，将 HDF5 数据复制到 Zarr 数据集
                            futures.add(
                                executor.submit(img_copy,
                                                img_arr, zarr_idx, hdf5_arr, hdf5_idx))

                    # 等待所有剩余的任务完成
                    completed, futures = concurrent.futures.wait(futures)
                    for f in completed:
                        if not f.result():  # 检查每个任务是否成功
                            raise RuntimeError('Failed to encode image!')
                    pbar.update(len(completed))  # 更新进度条

    # 创建一个 ReplayBuffer 对象，并将 root 传递给它作为存储的根目录
    replay_buffer = ReplayBuffer(root)

    # 返回创建的 ReplayBuffer 对象
    return replay_buffer


def normalizer_from_stat(stat):
    # 计算最大绝对值，取最大值的绝对值和最小值的绝对值中的最大值
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())

    # 创建一个和 stat['max'] 相同形状的数组，填充值为 1 / max_abs，表示归一化的缩放因子
    scale = np.full_like(stat['max'], fill_value=1 / max_abs)

    # 创建一个和 stat['max'] 相同形状的零数组，表示偏移量（通常用于数据的平移）
    offset = np.zeros_like(stat['max'])

    # 使用上面计算的 scale 和 offset 创建一个 SingleFieldLinearNormalizer 对象
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,  # 缩放因子
        offset=offset,  # 偏移量
        input_stats_dict=stat  # 输入的统计信息字典
    )

