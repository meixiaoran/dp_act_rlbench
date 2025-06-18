import numpy as np
from scipy.interpolate import interp1d

class Interpolator:
    """一个用于从 [batch_size, 2, 8] 到 [batch_size, target_samples, 8] 的线性插值类。"""
    
    def __init__(self, data, target_samples=10, kind='linear'):
        """
        初始化插值器。
        
        参数：
            data (np.ndarray): 输入数据，形状为 [batch_size, 2, 8]。
            target_samples (int): 目标样本数，默认为 10。
            kind (str): 插值方法，默认为 'linear'，支持 scipy.interpolate.interp1d 的所有方法。
        """
        # 验证输入数据形状
        if len(data.shape) != 3 or data.shape[1] != 2 or data.shape[2] != 8:
            raise ValueError("输入数据形状必须为 [batch_size, 2, 8]")
        
        self.data = data
        self.batch_size = data.shape[0]
        self.target_samples = target_samples
        self.kind = kind
        
        # 定义原始和目标索引
        self.x = np.array([0, 1])  # 原始 2 个样本的索引
        self.x_new = np.linspace(0, 1, target_samples)  # 目标样本的索引
        
        # 初始化输出数组
        self.interpolated_data = np.zeros((self.batch_size, target_samples, 8))
        
    def interpolate(self):
        """执行插值操作并返回结果。"""
        for b in range(self.batch_size):  # 遍历每个 batch
            for i in range(8):  # 遍历每个特征
                interp_func = interp1d(self.x, self.data[b, :, i], kind=self.kind)
                self.interpolated_data[b, :, i] = interp_func(self.x_new)
        
        return self.interpolated_data
    
    def get_interpolated_data(self):
        """获取插值结果。"""
        return self.interpolated_data
    
    def get_original_data(self, batch_idx=0, sample_idx=1):
        """获取原始数据中指定 batch 和样本的数据。"""
        if batch_idx >= self.batch_size or sample_idx >= 2:
            raise IndexError("batch_idx 或 sample_idx 超出范围")
        return self.data[batch_idx, sample_idx]
    
    def print_shapes(self):
        """打印原始数据和插值数据的形状。"""
        print("原始数据形状:", self.data.shape)
        print("插值后数据形状:", self.interpolated_data.shape)

# 示例用法
if __name__ == "__main__":
    # 假设输入数据
    data = np.random.rand(64, 2, 8)  # 形状为 [64, 2, 8]
    
    # 创建插值器实例
    interpolator = Interpolator(data, target_samples=10, kind='linear')
    
    # 执行插值
    interpolated_data = interpolator.interpolate()
    
    # 打印形状
    interpolator.print_shapes()
    
    