import h5py
from rlbench.const import SUPPORTED_ROBOTS
from RLBench_ACT.act.constants import DT
import numpy as np
from scipy.spatial.transform import Rotation
from RLBench_ACT.act.sim_env_rlbench import make_sim_env
import time
def compute_pose_in_world(T_BW, T_AB):
    # 从 T_BW 和 T_AB 中提取位置和四元数
    p_BW = np.array(T_BW[:3])  # B 在世界坐标系中的位置
    q_BW = np.array(T_BW[3:])  # B 在世界坐标系中的四元数

    p_AB = np.array(T_AB[:3])  # A 在 B 坐标系中的位置
    q_AB = np.array(T_AB[3:])  # A 在 B 坐标系中的四元数

    # 计算 A 在世界坐标系中的位置
    # 将四元数转换为旋转矩阵
    R_BW = Rotation.from_quat(q_BW).as_matrix()  # 从四元数生成旋转矩阵
    p_AW = R_BW @ p_AB + p_BW  # 世界坐标系中的位置

    # 计算 A 在世界坐标系中的姿态
    q_AW = Rotation.from_quat(q_BW) * Rotation.from_quat(q_AB)  # 四元数乘法
    q_AW = q_AW.as_quat()  # 转换为四元数格式

    # 返回结果：位置和四元数组合成 7 元素数组
    return np.concatenate([p_AW, q_AW])

task_name = 'reach_target'
onscreen_render = True
robot_name = 'panda'

env = make_sim_env(task_name, onscreen_render, robot_name)  # 创建模拟环境

robot_setup: str = 'panda'
robot_setup = robot_setup.lower()
arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
    robot_setup]
arm, gripper = arm_class(), gripper_class()
arm_pose = arm.get_pose()



with h5py.File(f"/home/rookie/桌面/diffusion_policy/data/dataset/close_laptop_lid/close_laptop_lid_merged_data.hdf5") as file:
    # count total steps
    demos = file['data']  # 打开 HDF5 文件并读取数据
    for i in range(56,100):
        env.reset()
        abort_variation = False
        demo = demos[f'demo_{i}']
        actions = demo['action']
        print(f"当前Demo{i}的动作步数为：" + str(actions.shape[0]))
        for action in actions:
            action_world = compute_pose_in_world(arm_pose, action[:-1])
            action_world = np.concatenate([action_world, np.array([action[-1]])])
            try:
                ts_obs, reward, terminate = env.step(action_world)
            except Exception as e:
                continue
        time.sleep(1)