import h5py
import re

def numeric_sort_key(filename):
    """
    自定义排序键，提取文件名中的数字部分进行排序。
    filename: 文件名（如 demo_1, demo_10 等）
    返回值: 提取出来的数字列表
    """
    return [int(s) for s in re.findall(r'\d+', filename)]

def print_hdf5_structure(group, indent=0, image_count=0):
    """
    递归打印 HDF5 文件的结构，并按数字顺序排序组名。同时统计图片数量。
    group: h5py.Group 或 h5py.File，HDF5 文件或组
    indent: 用于格式化输出的缩进级别
    image_count: 当前图片数量
    返回值: 累计的图片数量
    """
    # 遍历该组中的所有对象（组和数据集），并按数字顺序排序
    keys = sorted(group.keys(), key=numeric_sort_key)  # 按数字顺序排序
    for key in keys:
        item = group[key]
        # 根据对象类型进行处理
        if isinstance(item, h5py.Group):
            print('  ' * indent + f"{key}")  # 输出组的名字
            image_count = print_hdf5_structure(item, indent + 1, image_count)  # 递归调用，进入子组
        elif isinstance(item, h5py.Dataset):
            print('  ' * indent + f"{key}, {item.shape}, {item.dtype}")  # 输出数据集的信息
            # 如果数据集是图片数据（维度为 (N, H, W, C)），统计图片数量
            if len(item.shape) == 4 and item.shape[3] == 3:  # 判断是否是图像数据（假设最后一维为3，表示RGB）
                image_count += item.shape[0]  # 第0维是样本数量

    return image_count

def show_hdf5_structure_and_count_images(file_path):
    """
    打开 HDF5 文件并展示其结构，同时统计所有图片的数量。
    file_path: HDF5 文件的路径
    """
    with h5py.File(file_path, 'r') as f:
        print(f"打开 HDF5 文件: {file_path}")
        total_images = print_hdf5_structure(f)  # 打印文件的结构并统计图片数量
        print(f"总共图片数量: {total_images}")


# 使用示例
file_path = 'RLBench_ACT/Datasets/reach_target/variation0/reach_target_abs_merged_data.hdf5'  # 替换为你合成的 HDF5 文件路径
show_hdf5_structure_and_count_images(file_path)
