from multiprocessing import Process, Manager  # 导入多进程模块
from pyrep.const import RenderMode  # 导入渲染模式
from rlbench.observation_config import ObservationConfig, CameraConfig  # 导入观察配置模块
from rlbench.action_modes.action_mode import MoveArmThenGripper  # 导入移动手臂和抓取的动作模式
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition, EndEffectorPoseViaIK, EndEffectorPoseViaPlanning  # 导入关节速度和关节位置动作模式
from rlbench.noise_model import GaussianNoise  # 导入高斯噪声模型
from rlbench.action_modes.gripper_action_modes import Discrete  # 导入离散的抓取动作模式
from rlbench.backend.utils import task_file_to_task_class  # 导入任务文件转换工具
from rlbench.environment import Environment  # 导入RLBench环境
import rlbench.backend.task as task  # 导入任务模块

import os, socket  # 导入操作系统和网络相关模块
import pickle  # 导入序列化模块
from rlbench.backend.const import *  # 导入常量
import numpy as np  # 导入NumPy模块

from absl import app  # 导入ABSL库
from absl import flags  # 导入ABSL的标志库
import h5py  # 导入HDF5文件操作库
import sys  # 导入系统库
sys.path.append(".")  # 将当前路径添加到系统路径中

from constants import SIM_TASK_CONFIGS  # 导入自定义的任务配置
import json  # 导入JSON模块

import cv2  # 导入OpenCV库，用于图像处理
from PIL import Image  # 导入PIL库，用于图像处理
from rlbench.backend import utils  # 导入RLBench工具库
import time  # 导入时间模块


# 设置环境变量，指定COPPELIASIM插件路径
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.expanduser('~/COPPELIASIM')

# 使用FLAGS管理命令行参数
FLAGS = flags.FLAGS


flags.DEFINE_string('save_path', 'RLBench_ACT/Datasets', 'Where to save the demos.')
flags.DEFINE_list('tasks', [], 'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128], 'The size of the images tp save.')  # 640, 480  [height x width]，coppliasim [width x height]
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'], 'The renderer to use. opengl does not include shadows, ' 'but is faster.')
flags.DEFINE_integer('processes', 1, 'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 50, 'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', 1, 'Number of variations to collect per task. -1 for all.')
flags.DEFINE_boolean('onscreen_render', False, 'if onscreen render.')
flags.DEFINE_string('robot', 'panda', 'which robot do you want use.')
flags.DEFINE_boolean('abs', False, 'when using end_pose move action relative or absolute.')
np.set_printoptions(linewidth=200)  # 设置NumPy的打印选项

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
import numpy as np
from scipy.spatial.transform import Rotation
from rlbench.const import SUPPORTED_ROBOTS
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

# 保存演示数据
def save_demo(demo, example_path, ex_idx):
    data_dict = {
        '/action': [],
        '/obs/wrist': [],
        '/obs/front': [],
        '/obs/qpos': [],
    }
    # max_timesteps = len(demo)

    state2action_step = 1

    prev_action = None
    threshold = 0.0002
    # 遍历每个观测，保存动作和图像，当需要绝对动作时对action和qpos做特殊处理
    for i, obs in enumerate(demo):
        if i >= state2action_step:

            # print(dir(obs))
            action = np.append(obs.gripper_pose, obs.gripper_open)
            if prev_action is not None:
                action_diff = np.linalg.norm(action - prev_action)  # 欧几里得距离
                if action_diff < threshold:
                    continue  # 跳过当前时间步的数据者
            data_dict['/action'].append(action)
            prev_action = action  # 更新前一个动作

        data_dict['/obs/wrist'].append(obs.wrist_rgb)  # 保存腕部相机图像
        data_dict['/obs/front'].append(obs.front_rgb)  # 保存头部相机图像
        qpos = np.append(obs.gripper_pose, obs.gripper_open)
        data_dict['/obs/qpos'].append(qpos)


    print("action", len(data_dict['/action']))
    print("action:" + str(data_dict['/action']))
    print("/obs/wrist", len(data_dict['/obs/wrist']))

    # 填充缺失的动作数据
    for idx in range(state2action_step):
        action = np.append(obs.gripper_pose, obs.gripper_open)
        data_dict['/action'].append(action)

    action_len = np.shape(action)[0]
    max_timesteps = len(data_dict['/action'])
    # 生成保存路径
    dataset_path = os.path.join(example_path, f'episode_{ex_idx}')
    check_and_make(example_path)  # 确保路径存在

    # 保存数据到HDF5文件
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        action = root.create_dataset('action', (max_timesteps, action_len))
        obs = root.create_group('obs')
        obs.create_dataset('wrist', (max_timesteps, 128, 128, 3), dtype='uint8', chunks=(1, 128, 128, 3))
        obs.create_dataset('front', (max_timesteps, 128, 128, 3), dtype='uint8', chunks=(1, 128, 128, 3))
        obs.create_dataset('qpos', (max_timesteps, action_len))


        # 将数据写入HDF5文件
        for name, array in data_dict.items():
            root[name][...] = array
            print("name", name)
            print("name", root[name])
        print("演示数据保存成功")


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """每个线程会选择一个任务和变体，然后收集该变体下所有的回合数据 (episodes_per_task)。"""

    # 初始化线程的随机种子
    np.random.seed(10)  # 设置随机种子为 10

    num_tasks = len(tasks)  # 任务总数
    img_size = list(map(int, FLAGS.image_size))  # 获取图像尺寸，并转换为整数
    obs_config = ObservationConfig()  # 创建一个新的观察配置

    # 设置观察配置，关闭不需要的设置
    obs_config.set_all(False)  # 关闭所有默认的配置
    obs_config.wrist_camera.set_all(True)  # 打开腕部相机的所有设置
    obs_config.front_camera.set_all(True)  # 打开头部相机的所有设置
    obs_config.set_all_low_dim(True)  # 设置为低维度的观察（只保留必要的数据）

    # 设置腕部和头部相机的图像尺寸
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size
    obs_config.wrist_camera.depth_in_meters = False  # 设置不使用深度信息

    # 根据渲染模式选择相应的渲染设置
    if FLAGS.renderer == 'opengl':
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL  # 使用OpenGL渲染模式
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    elif FLAGS.renderer == 'opengl3':
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL3  # 使用OpenGL3渲染模式
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    # 打印当前的渲染设置和机器人配置
    print(f'{FLAGS.onscreen_render=}, {FLAGS.robot=}')

    # 如果开启屏幕渲染，则设为False，表示开启图形界面模式，否则为True表示头less模式
    headless_val = False if FLAGS.onscreen_render else True

    # 初始化RLBench环境，配置动作模式和观察配置
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),  # 设置动作模式：先移动手臂，再抓取
        obs_config=obs_config,  # 设置观察配置
        headless=headless_val,  # 是否开启图形界面
        robot_setup=FLAGS.robot  # 设置机器人类型（例如：Sawyer）
    )

    rlbench_env.launch()  # 启动环境
    task_env = None  # 初始化任务环境为空
    tasks_with_problems = results[i] = ''  # 记录遇到问题的任务，并初始化

    while True:
        with lock:  # 使用锁来确保线程安全，获取当前任务和变体
            if task_index.value >= num_tasks:  # 如果已经处理完所有任务
                print('Process', i, 'finished')  # 输出进程完成的提示
                break

            my_variation_count = variation_count.value  # 获取当前变体计数
            t = tasks[task_index.value]  # 获取当前任务
            task_env = rlbench_env.get_task(t)  # 获取当前任务的环境

            # 获取任务的变体数量
            var_target = task_env.variation_count()

            # 如果用户指定了变体数量，限制最大变体数量
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)

            # 如果当前变体数量已达上限，则重置变体计数，并切换到下一个任务
            if my_variation_count >= var_target:
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1  # 增加变体计数
            if task_index.value >= num_tasks:  # 如果任务已经收集完毕
                print('Process', i, 'finished')  # 输出进程完成的提示
                break
            t = tasks[task_index.value]  # 获取下一个任务

        # 获取任务环境并设置变体
        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)  # 设置当前的变体
        descriptions, _ = task_env.reset()  # 重置任务环境并获取任务描述

        # 设置保存路径
        if FLAGS.variations == 1:
            variation_path = os.path.join(FLAGS.save_path, task_env.get_name(), VARIATIONS_FOLDER % 0)
        else:
            varitation_index = ""
            for i in range(1, FLAGS.variations + 1):
                varitation_index = varitation_index + str(i)
            variation_path = os.path.join(FLAGS.save_path, task_env.get_name(),
                                          VARIATIONS_FOLDER % int(varitation_index))

        check_and_make(variation_path)  # 创建路径（如果不存在）

        # 保存任务描述到文件
        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)  # 将任务描述序列化保存到文件

        abort_variation = False  # 用于标记是否跳过当前变体

        for ex_idx in range(FLAGS.episodes_per_task):  # 遍历每个任务的回合数
            print('Process', i, '// Task:', task_env.get_name(), '// Variation:', my_variation_count, '// Demo:',
                  ex_idx)

            attempts = 10  # 尝试收集演示数据的最大次数

            # 获取当前任务的配置
            task_config = SIM_TASK_CONFIGS[FLAGS.tasks[0]]
            episode_len = task_config['episode_len']  # 每回合的时长

            while attempts > 0:  # 如果还有尝试次数，继续尝试收集演示数据
                try:
                    # 获取演示数据
                    demo, = task_env.get_demos(amount=1, live_demos=True, episode_len=episode_len)
                except Exception as e:  # 如果出现异常，重试
                    attempts -= 1
                    if attempts > 0:
                        continue
                        # 如果所有尝试都失败，记录错误信息，并跳过当前任务/变体
                    problem = (
                            'Process %d failed collecting task %s (variation: %d, '
                            'example: %d). Skipping this task/variation.\n%s\n' % (
                                i, task_env.get_name(), my_variation_count, ex_idx,
                                str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem  # 记录问题
                    abort_variation = True  # 设置为跳过当前变体
                    break

                # 成功获取到演示数据后，保存数据
                with file_lock:
                    save_demo(demo, variation_path, ex_idx + FLAGS.episodes_per_task * my_variation_count)
                break  # 跳出循环，表示成功获取到演示数据
            if abort_variation:  # 如果需要跳过当前变体，则退出
                break

    results[i] = tasks_with_problems  # 保存当前进程收集的数据状态
    rlbench_env.shutdown()  # 关闭RLBench环境


def main(argv):
    """主函数，负责初始化任务、配置多进程数据收集并启动并行数据收集过程。"""

    # 获取任务文件列表，排除 '__init__.py' 文件并去掉扩展名 '.py'
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    # 如果用户指定了任务列表，验证任务文件是否存在于当前任务列表中
    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)  # 如果有任务未找到，抛出错误
        task_files = FLAGS.tasks  # 如果有指定任务，则仅使用这些任务

    # 将任务文件转换为任务类，得到任务对象列表
    tasks = [task_file_to_task_class(t) for t in task_files]

    # 创建一个共享的Manager，用于跨进程共享数据
    manager = Manager()
    result_dict = manager.dict()  # 用于保存每个进程的结果（如任务处理状态、错误信息等）
    file_lock = manager.Lock()  # 用于保护文件写入操作的锁，确保多进程写文件时不会冲突
    task_index = manager.Value('i', 0)  # 用于跨进程共享当前任务的索引
    variation_count = manager.Value('i', 0)  # 用于跨进程共享当前任务变体的计数
    lock = manager.Lock()  # 用于保护任务和变体选择的锁，确保每个进程顺序执行

    # 检查并创建保存数据的目录（如果不存在的话）
    check_and_make(FLAGS.save_path)

    # 创建多个进程，任务分配到不同进程
    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]  # 通过循环创建多个进程，每个进程处理任务的部分

    # 启动所有进程
    [t.start() for t in processes]  # 启动每个进程

    # 等待所有进程结束
    [t.join() for t in processes]  # 等待所有进程执行完毕

    # 所有数据收集完成后的提示
    print('Data collection done!')

    # 打印每个进程的结果（处理的任务和变体，是否遇到问题等）
    for i in range(FLAGS.processes):
        print(result_dict[i])  # 输出每个进程的处理状态和问题（如果有的话）

        
if __name__ == '__main__':
  app.run(main)
