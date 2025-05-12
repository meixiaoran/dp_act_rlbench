"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform


OmegaConf.register_new_resolver("eval", eval, replace=True)

import click

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
# 参数 --input 或 -i，必须指定，用于传入模型检查点的路径（文件位置）。这个检查点通常是训练好的模型文件。
@click.option('--output', '-o', required=True, help='Directory to save recording')
# 参数 --output 或 -o，必须指定，用于指定录制结果保存的目录路径。
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
# 参数 --robot_ip 或 -ri，必须指定，表示 UR5 机械臂的 IP 地址，例如 192.168.0.204。用于与机器人进行网络通信。
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
# 参数 --match_dataset 或 -m，默认为 None。用于指定一个数据集，帮助覆盖并调整初始条件。
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
# 参数 --match_episode 或 -me，默认为 None（可以不指定），类型为整数。用于指定要从匹配数据集中选择的具体 episode（场景或序列）。
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
# 参数 --vis_camera_idx，默认为 0，类型为整数。用于指定要可视化的 RealSense 摄像头索引号。
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
# 参数 --init_joints 或 -j，这是一个布尔值选项（标志）。如果指定了该标志，表示程序开始时是否初始化机械臂的关节配置。默认值为 False（不初始化）。
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
# 参数 --steps_per_inference 或 -si，默认为 6，类型为整数。表示每次推理时的动作步长，即动作的时间跨度。
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
# 参数 --max_duration 或 -md，默认为 60，类型为浮点数。表示每个运行周期（epoch）的最大持续时间，以秒为单位。
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
# 参数 --frequency 或 -f，默认为 10，类型为浮点数。表示控制频率，以赫兹（Hz）为单位。决定机器人执行命令的频率。
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
# 参数 --command_latency 或 -cl，默认为 0.01，类型为浮点数。表示从接收 SpaceMouse（太空鼠）命令到在机器人上执行命令之间的延迟时间（以秒为单位）。

def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints,
         steps_per_inference, max_duration,
         frequency, command_latency):
    # 1. 加载匹配数据集
    match_camera_idx = 0  # 设置匹配数据集的相机索引为 0
    episode_first_frame_map = dict()  # 用字典存储每个 episode 对应的第一帧
    if match_dataset is not None:
        # 如果提供了匹配数据集路径
        match_dir = pathlib.Path(match_dataset)  # 将路径转为 Path 对象
        match_video_dir = match_dir.joinpath('videos')  # 找到视频文件夹
        # 遍历每个视频文件夹（每个文件夹代表一个 episode）
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)  # 解析 episode 索引
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')  # 拼接视频文件路径
            if match_video_path.exists():  # 如果视频文件存在
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)  # 读取视频文件的第一帧
                episode_first_frame_map[episode_idx] = frames[0]  # 将第一帧存入字典
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    # 打印加载了多少个 episode 的第一帧

    # 2. 加载模型检查点
    ckpt_path = input  # 从命令行参数获取检查点路径
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)  # 加载检查点文件
    cfg = payload['cfg']  # 从检查点中获取配置
    cls = hydra.utils.get_class(cfg._target_)  # 根据配置文件获取对应的类
    workspace = cls(cfg)  # 创建工作空间对象
    workspace: BaseWorkspace  # 明确指定工作空间的类型
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # 加载模型权重和配置
    print(dir(cfg))
    # 3. 根据模型类型设置特定参数
    action_offset = 0  # 动作偏移量初始化为 0
    delta_action = False  # 初始化 delta_action 标志为 False

    if 'diffusion' in cfg.name:
        # 如果模型是 diffusion 模型（基于扩散模型）
        policy: BaseImagePolicy
        policy = workspace.model  # 获取工作空间中的模型
        if cfg.training.use_ema:
            policy = workspace.ema_model  # 如果使用 EMA（指数滑动平均）模型，切换为 EMA 模型

        device = torch.device('cuda')  # 使用 CUDA 设备（GPU）
        policy.eval().to(device)  # 设置模型为评估模式，并转移到 GPU

        # 设置推理参数
        policy.num_inference_steps = 16  # 设置 DDIM 推理步骤数
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1  # 设置动作步骤数

    elif 'robomimic' in cfg.name:
        # 如果是 robomimic 模型（基于 BCRNN）
        policy: BaseImagePolicy
        policy = workspace.model  # 获取模型

        device = torch.device('cuda')  # 使用 CUDA 设备（GPU）
        policy.eval().to(device)  # 设置为评估模式并转移到 GPU

        # 对于 BCRNN，动作的步数始终为 1
        steps_per_inference = 1  # 设置每次推理的步骤数为 1
        action_offset = cfg.n_latency_steps  # 设置动作的偏移量为训练时延迟的步数
        delta_action = cfg.task.dataset.get('delta_action', False)  # 获取是否启用增量动作

    elif 'ibc' in cfg.name:
        # 如果是 ibc 模型（基于强化学习的模型）
        policy: BaseImagePolicy
        policy = workspace.model  # 获取模型
        policy.pred_n_iter = 5  # 设置预测迭代次数
        policy.pred_n_samples = 4096  # 设置每次预测的样本数量

        device = torch.device('cuda')  # 使用 CUDA 设备（GPU）
        policy.eval().to(device)  # 设置为评估模式并转移到 GPU
        steps_per_inference = 1  # 每次推理的步骤数为 1
        action_offset = 1  # 设置动作偏移量为 1
        delta_action = cfg.task.dataset.get('delta_action', False)  # 获取是否启用增量动作
    else:
        # 如果模型类型不匹配，抛出错误
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # 4. 设置实验参数
    dt = 1 / frequency  # 计算每次控制的时间间隔

    # 获取任务的观察分辨率
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps  # 获取观察步数
    print("n_obs_steps: ", n_obs_steps)  # 打印观察步数
    print("steps_per_inference:", steps_per_inference)  # 打印每次推理的步数


    with SharedMemoryManager() as shm_manager:
        # 使用 SharedMemoryManager 创建一个共享内存管理器。该管理器确保多个进程可以访问共享内存区域。

        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_image_resolution=obs_res,
                obs_float32=True,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # 为每个相机视图设置视频录制线程数 (H.264 编码)
                thread_per_video=3,
                # 设置视频质量，数值越低质量越差，但速度更快（例如 21 是一种折中的选择）
                video_crf=21,
                shm_manager=shm_manager) as env:
            # 使用 Spacemouse 和 RealEnv 作为上下文管理器，确保它们在代码块结束后自动关闭。

            cv2.setNumThreads(1)
            # 设置 OpenCV 使用的线程数为 1，避免多线程造成不必要的资源竞争。

            # 配置 RealSense 相机的参数
            env.realsense.set_exposure(exposure=120, gain=0)
            # 设置 RealSense 相机的曝光值为 120，增益值为 0。

            env.realsense.set_white_balance(white_balance=5900)
            # 设置 RealSense 相机的白平衡为 5900。

            print("Waiting for realsense")
            time.sleep(1.0)
            # 等待 1 秒钟，确保 RealSense 相机准备好。

            print("Warming up policy inference")
            # 打印信息表示正在热身策略推理。

            obs = env.get_obs()
            # 获取环境的观察数据（通常是图像或状态信息），这里通过 `env.get_obs()` 获取当前观察。

            with torch.no_grad():
                # 在推理阶段，禁用梯度计算以节省内存和计算资源。

                policy.reset()
                # 重置策略网络（如果需要的话），例如重置内部状态或模型参数。

                # 将环境的观察数据（`obs`）转换为模型所需要的输入格式
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                # 将环境的原始观察数据（`obs`）转换为符合模型要求的字典格式，形状元数据从配置中提取。

                # 使用字典映射，将观察数据转换为 PyTorch 张量，并将其移到指定的设备（GPU 或 CPU）。
                obs_dict = dict_apply(obs_dict_np,
                                      lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                # `dict_apply` 函数用于将 numpy 数组转换为张量，并添加批量维度。

                # 使用策略模型推理动作
                result = policy.predict_action(obs_dict)
                # 使用当前的观察数据输入到策略模型中，返回动作结果。

                action = result['action'][0].detach().to('cpu').numpy()
                # 获取模型输出的动作数据，`.detach()` 用于移除计算图，`to('cpu')` 将张量转移到 CPU，
                # `numpy()` 用于转换为 NumPy 数组，方便后续处理。

                assert action.shape[-1] == 7


                del result


            print('Ready!')
            # 打印 "Ready!"，表示所有准备工作完成，可以开始执行任务了。

            while True:
                # ========= 人类控制循环 ==========
                print("Human in control!")
                # 打印提示信息，表示当前是由人类操作控制机械臂。

                state = env.get_robot_state()
                # 获取当前机械臂的状态，包括位姿信息、关节状态等。

                target_pose = state['TargetTCPPose']
                # 提取目标 TCP（工具中心点）位姿，作为操作的初始目标。

                t_start = time.monotonic()
                # 获取当前时间，作为循环的起始时间。`time.monotonic()` 是一个单调计时器，不受系统时间变化影响。

                iter_idx = 0
                # 初始化循环的迭代计数器。

                while True:
                    # ===== 每次控制循环的操作 =====

                    # 1. 计算时间
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    # 当前循环的结束时间点，根据频率计算得出。

                    t_sample = t_cycle_end - command_latency
                    # 采样时间，考虑了命令的通信延迟。

                    t_command_target = t_cycle_end + dt
                    # 下一个控制周期的目标时间。

                    # 2. 获取当前的观察
                    obs = env.get_obs()
                    # 从环境中获取当前的观察数据（例如相机图像、传感器数据等）。

                    # 3. 可视化处理
                    episode_id = env.replay_buffer.n_episodes
                    # 获取当前 episode 的编号。

                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    # 提取当前相机的最后一帧图像，用于显示。

                    match_episode_id = episode_id
                    # 默认匹配的 episode ID 是当前 episode。
                    if match_episode is not None:
                        match_episode_id = match_episode
                        # 如果指定了匹配的 episode，使用指定的 episode ID。

                    if match_episode_id in episode_first_frame_map:
                        # 如果匹配的 episode 存在于第一帧字典中：
                        match_img = episode_first_frame_map[match_episode_id]
                        # 获取对应的匹配图像。

                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        # 获取匹配图像和可视化图像的分辨率。

                        tf = get_image_transform(
                            input_res=(iw, ih),
                            output_res=(ow, oh),
                            bgr_to_rgb=False)
                        # 生成图像变换函数，将匹配图像调整到与可视化图像一致的分辨率。

                        match_img = tf(match_img).astype(np.float32) / 255
                        # 应用变换并归一化图像。

                        vis_img = np.minimum(vis_img, match_img)
                        # 将两张图像叠加显示（取最小值以增强匹配部分）。

                    text = f'Episode: {episode_id}'
                    # 显示当前 episode 的编号。

                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255)
                    )
                    # 在图像上添加文本信息。

                    cv2.imshow('default', vis_img[..., ::-1])
                    # 显示图像窗口，将 BGR 格式转换为 RGB。

                    key_stroke = cv2.pollKey()
                    # 检测按键输入。

                    if key_stroke == ord('q'):
                        # 如果按下 'q' 键，退出程序。
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # 如果按下 'c' 键，退出人类控制模式，将控制权交给策略。
                        break

                    precise_wait(t_sample)
                    # 等待采样时间，确保控制周期的一致性。

                    # 4. 获取遥控命令
                    sm_state = sm.get_motion_state_transformed()
                    # 从空间鼠标获取运动状态，并进行必要的变换。

                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                    # 计算位置增量，根据空间鼠标的输入状态计算位移。

                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                    # 计算旋转增量。

                    if not sm.is_button_pressed(0):
                        # 如果按键 0 未被按下，进入平移模式：
                        drot_xyz[:] = 0
                        # 禁止旋转，保留平移。

                    else:
                        dpos[:] = 0
                        # 如果按键 0 被按下，禁止平移，仅允许旋转。

                    if not sm.is_button_pressed(1):
                        # 如果按键 1 未被按下，进入 2D 平移模式：
                        dpos[2] = 0
                        # 禁止 Z 轴的位移。

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    # 将旋转增量转换为旋转向量表示。

                    target_pose[:3] += dpos
                    # 更新目标位姿的平移部分。

                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    # 更新目标位姿的旋转部分，通过复合旋转计算新的方向。

                    target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40])
                    # 限制目标位姿的 X 和 Y 坐标在指定范围内，避免超出操作空间。

                    # 5. 执行遥控命令
                    env.exec_actions(
                        actions=[target_pose],
                        timestamps=[t_command_target - time.monotonic() + time.time()])
                    # 将目标位姿命令发送给机器人，并指定目标执行的时间戳。

                    precise_wait(t_cycle_end)
                    # 等待当前控制周期结束时间，确保周期性控制。

                    iter_idx += 1
                    # 迭代计数器加 1，进入下一次循环。

                # ========== 策略控制循环 ==============
                try:
                    # 启动一个新的 episode
                    policy.reset()
                    # 重置策略模型，准备新的推理任务。

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    # 计算开始的延迟时间，确保系统的同步。

                    t_start = time.monotonic() + start_delay
                    # 获取当前的单调时间（不受系统时间改变影响），并加上延迟。

                    env.start_episode(eval_t_start)
                    # 启动一个新的 episode，开始与环境的交互。

                    # 等待 1/30 秒以获得最新的一帧图像，这样可以减少整体的延迟
                    frame_latency = 1 / 30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    # 等待一小段时间，确保环境准备好数据后再开始推理。

                    print("Started!")
                    # 输出提示信息，表示 episode 已启动。

                    iter_idx = 0
                    # 初始化迭代计数器。

                    term_area_start_timestamp = float('inf')
                    # 初始化一个变量来记录终止区域的开始时间，初始为无限大。

                    perv_target_pose = None
                    # 初始化上一次的目标位姿为 `None`，用于后续的增量计算。

                    while True:
                        # 主控制循环，每次迭代执行一次策略推理并执行相应动作。

                        # 计算当前控制周期结束的时间
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt
                        # `t_cycle_end` 表示当前控制周期的结束时间。

                        # 获取观察数据
                        print('get_obs')
                        obs = env.get_obs()
                        # 从环境中获取当前的观察数据（例如传感器数据、图像数据等）。

                        obs_timestamps = obs['timestamp']
                        # 提取当前观察数据的时间戳。

                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        # 打印观察数据的延迟，计算当前时间与最后一帧观察数据时间戳之间的差值。

                        # 运行推理模型
                        with torch.no_grad():
                            s = time.time()
                            # 记录推理开始的时间。

                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            # 获取一个包含实际观察数据的字典，并根据任务的形状元数据进行处理。

                            obs_dict = dict_apply(obs_dict_np,
                                                  lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            # 将字典中的每个观察数据转换为张量，并传输到 GPU。

                            result = policy.predict_action(obs_dict)
                            # 使用策略模型进行推理，预测出对应的动作。

                            action = result['action'][0].detach().to('cpu').numpy()
                            # 获取推理结果中的动作，并将其从 GPU 转移到 CPU，再转为 NumPy 数组。

                            print('Inference latency:', time.time() - s)
                            # 打印推理所花费的时间，计算推理过程的延迟。

                        # 将策略的动作转换为环境可以执行的动作
                        if delta_action:
                            assert len(action) == 1
                            # 如果是增量动作模式，确保只有一个动作。

                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            # 如果没有前一次的目标位姿，则使用当前观察中的末尾工具末端位姿。

                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0, 1]] += action[-1]
                            # 增量计算目标位姿，修改目标位姿的 X 和 Y 坐标。

                            perv_target_pose = this_target_pose
                            # 更新上一轮的目标位姿。

                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                            # 将目标位姿扩展为 2D 数组。

                        else:
                            # 如果不是增量动作模式，直接使用策略预测的动作生成目标位姿。
                            this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                            this_target_poses[:] = target_pose
                            this_target_poses[:, [0, 1]] = action
                            # 创建目标位姿数组，并将预测的动作应用到目标位姿的 X 和 Y 坐标上。

                        # 处理时间同步
                        # 确保每次动作的执行时间是根据控制周期计算的，避免超时或延迟。
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                                             ) * dt + obs_timestamps[-1]
                        # 计算每个动作的时间戳，确保时间顺序正确。

                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        # 如果当前时间加上执行延迟小于动作的时间戳，说明动作还没到执行的时间。

                        if np.sum(is_new) == 0:
                            # 如果所有动作的时间戳都已经超时，意味着超出了时间预算。
                            this_target_poses = this_target_poses[[-1]]
                            # 只保留最后一个动作，推迟执行到下一个可用的时间点。

                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            # 计算下一个可执行时间。

                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                            # 将新的时间戳分配给超时的动作。

                        else:
                            # 如果动作时间戳仍然有效，则只执行那些有效的动作。
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # 限制目标动作范围
                        this_target_poses[:, :2] = np.clip(
                            this_target_poses[:, :2], [0.25, -0.45], [0.77, 0.40])
                        # 对目标位姿的 X 和 Y 坐标进行范围限制，避免超出环境的有效区域。

                        # 执行动作
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        # 将目标动作和时间戳传递给环境，执行动作。

                        print(f"Submitted {len(this_target_poses)} steps of actions.")
                        # 打印已经提交的动作步骤数。

                        # ========== 可视化部分 ==========
                        episode_id = env.replay_buffer.n_episodes
                        # 获取当前回合的编号（从环境的 replay buffer 中获取）。

                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        # 获取当前回合的最后一帧图像，使用给定的摄像头索引（`vis_camera_idx`）。

                        # 在图像上添加文本信息，显示回合编号和当前时间
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255)
                        )
                        # 在图像的左上角添加回合信息，文本内容为回合编号和当前时长。

                        cv2.imshow('default', vis_img[..., ::-1])
                        # 显示图像，通过 OpenCV 的 `imshow` 函数展示当前图像（RGB 转换为 BGR 格式）。

                        key_stroke = cv2.pollKey()
                        # 检测键盘输入，如果按下键则获取按键的键值。

                        if key_stroke == ord('s'):
                            # 如果按下 's' 键，停止当前回合。
                            env.end_episode()
                            print('Stopped.')
                            break
                        # 结束当前回合，并打印 'Stopped.'，然后退出循环。

                        # ========== 自动终止条件 ==========
                        terminate = False
                        # 初始化 `terminate` 为 `False`，表示是否需要终止当前回合。

                        # 超过最大时长后自动终止
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')
                        # 如果当前时间超过了最大允许时长 `max_duration`，则设置 `terminate` 为 `True`，并打印超时终止信息。

                        # 计算当前机器人末端执行器（EEF）位置与目标位置的距离（仅考虑位置的前两个坐标）
                        term_pose = np.array(
                            [3.40948500e-01, 2.17721816e-01, 4.59076878e-02, 2.22014183e+00, -2.22184883e+00,
                             -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        # 从环境中获取当前机器人的末端执行器位置（`robot_eef_pose`）。

                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        # 计算当前末端执行器位置和终止位置之间的距离，只计算 `x` 和 `y` 方向的距离。

                        # 如果机器人末端执行器位置与目标位置的距离小于 0.03，则进入终止区域
                        if dist < 0.03:
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        # 如果末端执行器进入了设定的终止区域（距离小于 0.03），并且在该区域停留超过 0.5 秒，则终止当前回合。
                        else:
                            # 如果末端执行器不在终止区域，则重置终止区域的时间戳。
                            term_area_start_timestamp = float('inf')

                        # 如果触发终止条件（超时或进入终止区域），则结束当前回合。
                        if terminate:
                            env.end_episode()
                            break
                        # 结束回合并退出循环。

                        # ========== 等待执行的同步 ==========
                        precise_wait(t_cycle_end - frame_latency)
                        # 等待当前控制周期结束前的时间（考虑到帧延迟），确保系统按时执行操作。

                        iter_idx += steps_per_inference
                        # 更新迭代索引，准备进行下一轮推理，步数按 `steps_per_inference` 递增。



                except KeyboardInterrupt:

                    print("Interrupted!")

                    # 捕获 KeyboardInterrupt 异常（通常是用户按下 Ctrl+C 触发）。

                    # 当用户通过键盘中断程序时，会打印 "Interrupted!" 信息，表示程序被手动停止。

                    # 停止机器人执行，并结束当前回合。

                    env.end_episode()

                    # 调用 `env.end_episode()` 结束当前的回合，停止机器人的运动。

                print("Stopped.")

                # 打印 "Stopped."，表示程序已经停止。


# %%
if __name__ == '__main__':
    main()
