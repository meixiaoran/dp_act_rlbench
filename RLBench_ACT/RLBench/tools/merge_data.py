import os
import re
import h5py


def numeric_sort_key(filename):
    """
    自定义排序键，提取文件名中的数字部分进行排序。
    filename: 文件名
    返回值: 提取出来的数字列表
    """
    # 使用正则表达式提取文件名中的数字部分
    return [int(s) for s in re.findall(r'\d+', filename)]


def merge_demos(input_dir, output_path):
    """
    合并多个 HDF5 文件中的数据到一个新的文件
    input_dir: 存放所有 episode_*.hdf5 文件的目录
    output_path: 输出合并后的 HDF5 文件路径
    """
    # 创建一个新的 HDF5 文件来保存合并后的数据
    with h5py.File(output_path, 'w') as root:
        root.attrs['sim'] = True  # 设置一个全局属性，标明是仿真数据

        # 创建 /data 组来存放所有演示
        data_group = root.create_group('data')

        # 获取所有 episode_*.hdf5 文件，按数字顺序排序
        episode_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.hdf5') and f.startswith('episode_')],
                               key=numeric_sort_key)  # 使用自定义排序键
        print(episode_files)

        # 遍历所有的 episode_*.hdf5 文件
        for episode_idx, filename in enumerate(episode_files):
            print("episode_idx", episode_idx)
            print("filename", filename)
            # 构造每个文件的完整路径
            file_path = os.path.join(input_dir, filename)

            # 打开每个 HDF5 文件
            with h5py.File(file_path, 'r') as f:
                # 提取当前文件中的数据
                try:
                    action_data = f['/action'][:]
                except KeyError:
                    print(f"警告: {filename} 中没有 '/action' 数据集")
                    continue  # 跳过这个文件，继续下一个文件

                # 尝试读取 '/obs/wrist' 和 '/obs/head' 数据集，如果没有则跳过
                wrist_data = f.get('/obs/wrist', None)
                head_data = f.get('/obs/front', None)
                qpos_data = f.get('/obs/qpos', None)
                print('action_data', action_data)
                print('qposdata', qpos_data[:] )

                # 检查是否存在 wrist 和 head 数据
                if wrist_data is None or head_data is None or qpos_data is None:
                    print(f"警告: {filename} 中缺少 '/obs/wrist' 或 '/obs/head' 或'/obs/qpos'数据集")
                    continue  # 跳过这个文件，继续下一个文件

                wrist_data = wrist_data[:]
                head_data = head_data[:]
                qpos_data = qpos_data[:]

                # 为每个 episode 创建一个新的组
                demo_group = data_group.create_group(f'demo_{episode_idx}')

                # 创建 /demo_{episode_idx}/obs 子组并保存 wrist 和 head 数据
                obs_group = demo_group.create_group('obs')
                obs_group.create_dataset('wrist', data=wrist_data, dtype='uint8', chunks=(1, 128, 128, 3))
                obs_group.create_dataset('front', data=head_data, dtype='uint8', chunks=(1, 128, 128, 3))
                obs_group.create_dataset('qpos', data=qpos_data, dtype='float32')

                # 创建 /demo_{episode_idx}/action 数据集
                demo_group.create_dataset('action', data=action_data, dtype='float32')

                # 打印日志
                print(f"成功合并 {filename} 到 /data/demo_{episode_idx}")

        print(f"所有演示数据已合并并保存到 {output_path}")


# 使用示例
input_dir = '/home/rookie/桌面/diffusion_policy/RLBench_ACT/Datasets/close_laptop_lid/variation0'  # 你的 HDF5 文件所在目录
output_path = '/home/rookie/桌面/diffusion_policy/data/dataset/close_laptop_lid/close_laptop_lid_merged_data.hdf5'  # 合并后的输出文件
merge_demos(input_dir, output_path)
