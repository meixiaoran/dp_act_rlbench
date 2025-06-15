from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 sequence_length: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=dict(),
                 episode_mask: Optional[np.ndarray] = None,
                 ):
        """
        初始化序列采样器。

        参数:
        replay_buffer: ReplayBuffer 实例，用于存储和访问数据。
        sequence_length: int, 期望的序列长度。
        pad_before: int, 在序列前填充的步骤数量。
        pad_after: int, 在序列后填充的步骤数量。
        keys: 用于从重放缓冲区中选择特定数据的键的列表。如果未提供，默认选择所有键。
        key_first_k: dict, 键为数据键，值为整数，表示只从每个键的前 k 个数据中采样，以提高性能。
        episode_mask: np.ndarray, 可选的布尔数组，用于指定哪些片段是有效的采样候选。

        方法:
        assert(sequence_length >= 1) 确保序列长度至少为 1。
        """

        super().__init__()
        # 检查 keys 参数是否为 None，如果是，则从 replay_buffer 中获取所有键。
        if keys is None:
            keys = list(replay_buffer.keys())

        # 获取重放缓冲区中的片段结束位置。
        episode_ends = replay_buffer.episode_ends[:]

        # 如果没有提供 episode_mask，则创建一个所有元素为 True 的掩码。
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        # 根据提供的掩码和其他参数创建采样索引。
        if np.any(episode_mask):
            indices = create_indices(episode_ends,
                                     sequence_length=sequence_length,
                                     pad_before=pad_before,
                                     pad_after=pad_after,
                                     episode_mask=episode_mask
                                     )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # 存储采样索引、键列表、序列长度和重放缓冲区引用。
        self.indices = indices
        self.keys = list(keys)  # 防止 OmegaConf 列表性能问题
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        # 返回可采样序列的总数。
        return len(self.indices)

    def sample_sequence(self, idx):
        # 根据指定索引采样序列。
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # 性能优化，避免可能的小内存分配。
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # 性能优化，只加载使用的观测步。
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # 使用 NaN 填充值以捕获错误。
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
                except Exception as e:
                    import pdb;
                    pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                # 如果需要填充。
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
