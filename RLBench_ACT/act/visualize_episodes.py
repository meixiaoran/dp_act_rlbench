import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython

e = IPython.embed

# 定义关节名称
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# 定义状态名称，状态名称是关节名称加上“gripper”
STATE_NAMES = JOINT_NAMES + ["gripper"]


# 加载 HDF5 数据集
def load_hdf5(dataset_dir, dataset_name):
    """
    从指定的目录加载 HDF5 数据集。

    参数:
    - dataset_dir: 数据集存放的目录。
    - dataset_name: 数据集文件名（不包括 .hdf5 后缀）。

    返回:
    - qpos: 机器人关节位置的数组。
    - action: 机器人执行的动作数组。
    - image_dict: 包含不同摄像头图像数据的字典。
    """
    # 拼接数据集路径
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')

    # 检查文件是否存在
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    # 打开 HDF5 文件并读取数据
    with h5py.File(dataset_path, 'r') as root:
        print("HDF5 file structure:")
        print("Root structure:", list(root.keys()))  # 打印根目录的所有组/数据集

        # 打印每个组的内容
        for group_name in root:
            print(f"Group: {group_name}")
            item = root[group_name]
            if isinstance(item, h5py.Group):
                # 如果是组，列出该组下的子项
                print("  Group contents:", list(item.keys()))
                for sub_group_name in item:
                    sub_group = item[sub_group_name]
                    if isinstance(sub_group, h5py.Group):
                        print(f"    Sub-group: {sub_group_name}")
                        print(f"      Sub-group contents: {list(sub_group.keys())}")
                    elif isinstance(sub_group, h5py.Dataset):
                        print(f"    Dataset: {sub_group_name}, Shape: {sub_group.shape}, dtype: {sub_group.dtype}")
            elif isinstance(item, h5py.Dataset):
                # 如果是数据集，打印数据集的名称、形状和数据类型
                print(f"  Dataset: {group_name}, Shape: {item.shape}, dtype: {item.dtype}")

        # 具体读取数据
        is_sim = root.attrs['sim']  # 检查是否为仿真数据
        print(f"Simulation data: {is_sim}")
        qpos = root['/observations/qpos'][()]  # 获取关节位置数据
        print(f"qpos shape: {qpos.shape}, dtype: {qpos.dtype}")
        action = root['/action'][()]  # 获取动作数据
        print(f"action shape: {action.shape}, dtype: {action.dtype}")
        image_dict = dict()  # 初始化图像字典

        # 遍历所有摄像头图像数据
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]  # 保存每个摄像头的图像数据
            print(f"Camera: {cam_name}, Image shape: {image_dict[cam_name].shape}, dtype: {image_dict[cam_name].dtype}")

    # 返回关节位置、动作和图像数据
    return qpos, action, image_dict


# 主函数
def main(args):
    """
    主程序，负责加载数据集、保存视频和可视化关节数据。

    参数:
    - args: 命令行参数，包含数据集目录、数据集索引等信息。
    """
    dataset_dir = args['dataset_dir']  # 获取数据集目录
    episode_idx = args['episode_idx']  # 获取数据集的 Episode 索引
    dataset_name = f'episode_{episode_idx}'  # 拼接数据集文件名

    # 加载数据集
    qpos, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    # 保存视频文件
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    # 可视化关节状态和命令
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))


# 保存视频文件
def save_videos(video, dt, video_path=None):
    """
    将视频保存为 mp4 文件。

    参数:
    - video: 图像数据，可能是字典或列表格式。
    - dt: 时间步长，用于计算视频的帧率。
    - video_path: 保存视频的路径。
    """
    if isinstance(video, list):  # 如果视频是一个列表（多个时间步的数据）
        cam_names = list(video[0].keys())  # 获取摄像头名称
        h, w, _ = video[0][cam_names[0]].shape  # 获取图像的高度和宽度
        w = w * len(cam_names)  # 将图像宽度乘以摄像头数量
        fps = int(1 / dt)  # 计算帧率
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 初始化视频写入器
        # 遍历每一帧图像
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:  # 获取每个摄像头的图像数据
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # 调整颜色通道（BGR 转 RGB）
                images.append(image)
            images = np.concatenate(images, axis=1)  # 将不同摄像头的图像水平拼接
            out.write(images)  # 写入视频帧
        out.release()  # 释放视频写入器
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):  # 如果视频是一个字典（单一时间步的数据）
        cam_names = list(video.keys())  # 获取摄像头名称
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])  # 获取所有摄像头的图像数据
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # 按宽度维度拼接所有摄像头的图像

        n_frames, h, w, _ = all_cam_videos.shape  # 获取视频的帧数、高度和宽度
        fps = int(1 / dt)  # 计算帧率
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 初始化视频写入器
        # 遍历每一帧图像
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # 调整颜色通道（BGR 转 RGB）
            out.write(image)  # 写入视频帧
        out.release()  # 释放视频写入器
        print(f'Saved video to: {video_path}')


# 可视化关节位置和命令
def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    """
    绘制关节位置（qpos）和命令（action）的曲线图。

    参数:
    - qpos_list: 关节位置数据列表。
    - command_list: 动作命令数据列表。
    - plot_path: 保存图表的路径。
    - ylim: y 轴的显示范围（可选）。
    - label_overwrite: 标签的覆盖内容（可选）。
    """
    if label_overwrite:
        label1, label2 = label_overwrite  # 如果提供了自定义标签，则使用自定义标签
    else:
        label1, label2 = 'State', 'Command'  # 默认标签

    qpos = np.array(qpos_list)  # 将关节位置列表转换为数组
    command = np.array(command_list)  # 将动作命令列表转换为数组
    num_ts, num_dim = qpos.shape  # 获取时间步数和维度数
    h, w = 2, num_dim  # 图表的高度和宽度
    num_figs = num_dim  # 需要绘制的图表数量
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))  # 创建子图

    # 绘制关节状态
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)  # 绘制关节位置
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')  # 设置标题
        ax.legend()

    # 绘制动作命令
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)  # 绘制动作命令
        ax.legend()

    # 设置 y 轴范围
    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()  # 自动调整布局
    plt.savefig(plot_path)  # 保存图表
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()  # 关闭图表


# 可视化时间戳（暂时没有用到）
def visualize_timestamp(t_list, dataset_path):
    """
    绘制时间戳的变化。

    参数:
    - t_list: 时间戳列表。
    - dataset_path: 数据集路径。
    """
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')  # 设置图表保存路径
    h, w = 4, 10  # 设置图表的高度和宽度
    fig, axs = plt.subplots(2, 1, figsize=(w, h * 2))  # 创建子图
    # 将时间戳转换为浮动秒数
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)  # 绘制时间戳变化
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float) - 1), t_float[:-1] - t_float[1:])  # 绘制时间差
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()  # 自动调整布局
    plt.savefig(plot_path)  # 保存图表
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()  # 关闭图表


# 运行主程序
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
