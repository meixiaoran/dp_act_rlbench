
# import asyncio
# import websockets
# from openpi_client import msgpack_numpy
# import numpy as np
# import time
# import hashlib
# from RLBench_ACT.act.sim_env_rlbench import make_sim_env
# from scipy.spatial.transform import Rotation

# def compute_pose_in_world(T_BW, T_AB):
#     p_BW = np.array(T_BW[:3])
#     q_BW = np.array(T_BW[3:])
#     p_AB = np.array(T_AB[:3])
#     q_AB = np.array(T_AB[3:])
#     R_BW = Rotation.from_quat(q_BW).as_matrix()
#     p_AW = R_BW @ p_AB + p_BW
#     q_AW = (Rotation.from_quat(q_BW) * Rotation.from_quat(q_AB)).as_quat()
#     return np.concatenate([p_AW, q_AW])

# async def run_client():
#     uri = "ws://localhost:8000"
#     task_name = "pick_up_cup"
#     robot_name = "panda"
#     max_episodes = 5
#     max_steps = 300
#     batch_size = 1
#     max_size = 2**26  # 64MB 以防止消息截断

#     try:
#         env = make_sim_env(task_name, onscreen_render=True, robot_name=robot_name)
#         env.set_variation(0)
#         print("[DEBUG] 环境初始化成功")
#     except Exception as e:
#         print(f"[ERROR] 环境初始化失败: {e}")
#         return

#     episode_returns = []
#     highest_rewards = []
#     packer = msgpack_numpy.Packer()

#     try:
#         async with websockets.connect(uri, compression=None, max_size=max_size, ping_interval=None, ping_timeout=None) as ws:
#             for ep in range(max_episodes):
#                 print(f"\n=== Episode {ep + 1} ===")
#                 rewards = []
#                 try:
#                     _, ts_obs = env.reset()
#                     print("[DEBUG] 环境重置成功")
#                 except Exception as e:
#                     print(f"[ERROR] 环境重置失败: {e}")
                   
#                     continue

#                 done = np.array([False] * batch_size)
#                 t = 0
#                 if ts_obs.gripper_pose.shape != (7,):
#                     print(f"[ERROR] gripper_pose 形状错误: 期望 (7,), 实际 {ts_obs.gripper_pose.shape}")
#                     continue
#                 arm_pose = ts_obs.gripper_pose

#                 # 验证 ts_obs 字段
#                 required_fields = ["wrist_rgb", "front_rgb", "wrist_point_cloud", "front_point_cloud", "gripper_pose", "gripper_open"]
#                 for field in required_fields:
#                     if not hasattr(ts_obs, field):
#                         print(f"[ERROR] ts_obs 缺少字段: {field}")
#                         break
#                 else:
#                     while not np.all(done) and t < max_steps:
#                         try:
#                             # 构造 state 为 float32
#                             state_np = np.append(ts_obs.gripper_pose, ts_obs.gripper_open).astype(np.float32)[np.newaxis]
#                             print(f"[DEBUG] wrist_rgb: shape={ts_obs.wrist_rgb.shape}, dtype={ts_obs.wrist_rgb.dtype}")

#                             # 构造 observation（保持原始结构）
#                             obs = {
#                                 "task": task_name,
#                                 "wrist_image": ts_obs.wrist_rgb[np.newaxis],
#                                 "front_image": ts_obs.front_rgb[np.newaxis],
#                                 "wrist_point_cloud": ts_obs.wrist_point_cloud.astype(np.float32)[np.newaxis],
#                                 "front_point_cloud": ts_obs.front_point_cloud.astype(np.float32)[np.newaxis],
#                                 "state": state_np,
#                             }
#                             print(f"[DEBUG] obs.wrist_image: shape={obs['wrist_image'].shape}, dtype={obs['wrist_image'].dtype}")
#                             # obs_size = sum(v.nbytes for v in obs.values() if isinstance(v, np.ndarray))
#                             # print(f"[DEBUG] 观测数据大小: {obs_size / 1024**2:.2f} MB")

#                             # if obs_size > max_size:
#                             #     print(f"[ERROR] 观测数据大小 ({obs_size / 1024**2:.2f} MB) 超过 max_size")
                                
#                             #     break

#                             packed_obs = packer.pack(obs)
#                             print("[DEBUG] 观测数据字典内容:")
#                             for key, value in obs.items():
#                                 if isinstance(value, np.ndarray):
#                                     print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
#                                 else:
#                                     print(f"  {key}: type={type(value)}")

#                             # obs_checksum = hashlib.sha256(packed_obs).hexdigest()
#                             # print(f"[DEBUG] 打包后的观测数据大小: {len(packed_obs) / 1024**2:.2f} MB, 校验和: {obs_checksum}")
#                             await ws.send(packed_obs)
#                             print("[DEBUG] 观测数据发送成功")
#                         except websockets.exceptions.ConnectionClosed as e:
#                             print(f"[ERROR] WebSocket 在发送观测数据时连接关闭: {e}")
#                             break
#                         except Exception as e:
#                             print(f"[ERROR] 发送观测数据失败: {e}")
#                             break

#                         try:
#                             action_bin = await ws.recv()
#                             print(f"[DEBUG] 收到动作消息: 长度={len(action_bin)} 字节")
#                             action_msg = msgpack_numpy.unpackb(action_bin, raw=False)
                            
#                             # 强制将其中的 numpy 数组变为可写副本
#                             if isinstance(action_msg, dict) and "action" in action_msg:
#                                 action_msg["action"] = np.array(action_msg["action"]).copy()

#                             action_msg = action_msg.copy()

#                             print(f"[DEBUG] 解包后的动作消息: {action_msg}")
#                             if not isinstance(action_msg, dict) or "action" not in action_msg:
#                                 print("[ERROR] 动作消息格式错误: 期望包含 'action' 键的字典")
#                                 # await ws.send(packer.pack({"error": "Invalid action message format"}))
#                                 break
#                             action = np.asarray(action_msg["action"])
#                             if action.shape != (batch_size, 8) or action.dtype != np.float32:
#                                 print(f"[ERROR] 动作无效: 形状 {action.shape}, 类型 {action.dtype}")
#                                 # await ws.send(packer.pack({"error": f"Invalid action: shape {action.shape}, dtype {action.dtype}"}))
#                                 # break
#                             action = action[0]
#                             # if not np.isclose(action[7], 0.0) and not np.isclose(action[7], 1.0):
#                             #     print(f"[ERROR] 无效的夹爪状态: {action[7]}")
#                             #     # await ws.send(packer.pack({"error": f"Invalid gripper state: {action[7]}"}))
#                             #     break
#                         except websockets.exceptions.ConnectionClosed as e:
#                             print(f"[ERROR] WebSocket 在接收动作时连接关闭: {e}")
#                             # break
#                         except Exception as e:
#                             print(f"[ERROR] 接收或解析动作失败: {e}")
#                             # break
#                         action[7] = 1.0 if action[7] > 0.95 else 0.0
#                         print(action)
#                         # quat = action[3:7]
#                         # quat /= np.linalg.norm(quat) + 1e-8
#                         # action[:7] = compute_pose_in_world(arm_pose, action[:3].tolist() + quat.tolist())
#                         # 四元数归一化
#                         quaternion_raw = action[3:7]
#                         norm = np.linalg.norm(quaternion_raw)
#                         if norm > 0:
#                             quaternion_normalized = quaternion_raw / norm
#                         else:
#                             quaternion_normalized = np.array([1.0, 0.0, 0.0, 0.0])
#                         action[3:7] = quaternion_normalized

#                         # 提取 gripper 控制值
#                         gripper = action[7]

#                         # 相对动作到绝对动作转换（前 7 维: xyz + 四元数）
#                         action_world = compute_pose_in_world(arm_pose, action[:7])

#                         # 拼接 gripper 值，构成完整的 8 维动作
#                         action = np.concatenate([action_world, [gripper]])
                      
# # 四元数归一化
                       
#                         reward = 0
#                         done = False
#                         success = 0
#                         try:
#                             ts_obs, reward, done = env.step(np.concatenate([action[:7], [action[7]]]))
#                             print(f"[DEBUG] 环境步进: 奖励={reward}, 完成={done}")
#                             if not isinstance(reward, (int, float, np.floating)):
#                                 print(f"[ERROR] 奖励类型无效: 期望标量，实际 {type(reward)}")
                                
#                                 # break
#                             if not isinstance(done, (bool, np.bool_)):
#                                 print(f"[ERROR] 完成状态类型无效: 期望布尔值，实际 {type(done)}")
                                
#                                 # break
#                         except Exception as e:
#                             print(f"[ERROR] 环境步进失败: {e}")
#                             # break

#                         success = reward == 1.0 and np.isclose(action[7], 1.0)
#                         feedback = {
#                             "observation": {
#                                 "task": task_name,
#                                 "wrist_image": ts_obs.wrist_rgb[np.newaxis],
#                                 "front_image": ts_obs.front_rgb[np.newaxis],
#                                 "wrist_point_cloud": ts_obs.wrist_point_cloud.astype(np.float32)[np.newaxis],
#                                 "front_point_cloud": ts_obs.front_point_cloud.astype(np.float32)[np.newaxis],
#                                 "state": np.append(ts_obs.gripper_pose, ts_obs.gripper_open).astype(np.float32)[np.newaxis],
#                             },
#                             "reward": float(reward),
#                             "terminated": bool(done),
#                             "success": bool(success),
#                         }
#                         print(f"[DEBUG] feedback['observation']['wrist_image']: type={type(feedback['observation']['wrist_image'])}, shape={feedback['observation']['wrist_image'].shape}")
#                         try:
#                             # feedback_size = sum(v.nbytes for v in feedback["observation"].values() if isinstance(v, np.ndarray))
#                             # print(f"[DEBUG] 反馈数据大小: {feedback_size / 1024**2:.2f} MB")
#                             # if feedback_size > max_size:
#                             #     print(f"[ERROR] 反馈数据大小 ({feedback_size / 1024**2:.2f} MB) 超过 max_size")
#                             #     await ws.send(packer.pack({"error": "Feedback size too large"}))
#                             #     break
#                             transition = (feedback["observation"], feedback["reward"], feedback["terminated"])
#                             packed_feedback = packer.pack(transition)
#                             # feedback_checksum = hashlib.sha256(packed_feedback).hexdigest()
#                             # print(f"[DEBUG] 打包后的反馈数据大小: {len(packed_feedback) / 1024**2:.2f} MB, 校验和: {feedback_checksum}")
#                             await ws.send(packed_feedback)
#                             print("[DEBUG] 反馈数据发送成功")
#                         except websockets.exceptions.ConnectionClosed as e:
#                             print(f"[ERROR] WebSocket 在发送反馈数据时连接关闭: {e}")
                            
#                         except Exception as e:
#                             print(f"[ERROR] 发送反馈数据失败: {e}")
                           

#                         rewards.append(reward)
#                         print(f"Step {t + 1} - 奖励: {reward}, 成功: {success}")
#                         if success:
#                             done = np.array([True] * batch_size)
#                         t += 1
#                         print(t)

#                 episode_return = np.sum(rewards)
#                 episode_returns.append(episode_return)
#                 highest_rewards.append(np.max(rewards) if rewards else 0.0)

#             success_rate = np.mean(np.array(highest_rewards) == 1.0)
#             avg_return = np.mean(episode_returns)
#             print(f"\n成功率: {success_rate * 100:.2f}%")
#             print(f"平均回报: {avg_return:.2f}")

#     except websockets.exceptions.ConnectionClosed as e:
#         print(f"[ERROR] WebSocket 意外关闭: {e}")
#     except Exception as e:
#         print(f"[ERROR] 意外错误: {e}")
#     finally:
#         pass

# if __name__ == "__main__":
#     asyncio.run(run_client())
import asyncio
import websockets
import msgpack_numpy
import numpy as np
import time
import hashlib
from RLBench_ACT.act.sim_env_rlbench import make_sim_env
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
async def run_client():
    uri = "ws://localhost:8000"
    task_name = "pick_up_cup"
    robot_name = "panda"
    max_episodes = 100
    max_steps = 200
    batch_size = 1
    max_size = 2**26  # 64MB 以防止消息截断

    try:
        env = make_sim_env(task_name, onscreen_render=True, robot_name=robot_name)
        env.set_variation(0)
        print("[DEBUG] 环境初始化成功")
    except Exception as e:
        print(f"[ERROR] 环境初始化失败: {e}")
        return

    episode_returns = []
    highest_rewards = []
    packer = msgpack_numpy.Packer()

    try:
        async with websockets.connect(uri, compression=None, max_size=max_size, ping_interval=None, ping_timeout=None) as ws:
            for ep in range(max_episodes):
                print(f"\n=== Episode {ep + 1} ===")
                rewards = []
                try:
                    _, ts_obs = env.reset()
                    print("-------------------环境重置成功---------------------")
                except Exception as e:
                    print(f"+++++++++++++环境重置失败: {e}+++++++++++++++")
                    continue

                done = False
                t = 0
                if ts_obs.gripper_pose.shape != (7,):
                    print(f"[ERROR] gripper_pose 形状错误: 期望 (7,), 实际 {ts_obs.gripper_pose.shape}")
                    continue
                # arm_pose = ts_obs.gripper_pose
                robot_setup: str = 'panda'
                robot_setup = robot_setup.lower()
                arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
                    robot_setup]
                arm, gripper = arm_class(), gripper_class()
                arm_pose = arm.get_pose()
                # 验证 ts_obs 字段
                required_fields = ["wrist_rgb", "front_rgb", "wrist_point_cloud", "front_point_cloud", "gripper_pose", "gripper_open"]
                for field in required_fields:
                    if not hasattr(ts_obs, field):
                        print(f"[ERROR] ts_obs 缺少字段: {field}")
                        break
                # 发送初始 observation
                try:
                    state_np = np.append(ts_obs.gripper_pose, ts_obs.gripper_open).astype(np.float32)[np.newaxis]
                    # print(f"[DEBUG] wrist_rgb: shape={ts_obs.wrist_rgb.shape}, dtype={ts_obs.wrist_rgb.dtype}")

                    obs = {
                        "task": task_name,
                        "wrist_image": ts_obs.wrist_rgb[np.newaxis],
                        "front_image": ts_obs.front_rgb[np.newaxis],
                        "wrist_point_cloud": ts_obs.wrist_point_cloud.astype(np.float32)[np.newaxis],
                        "front_point_cloud": ts_obs.front_point_cloud.astype(np.float32)[np.newaxis],
                        "state": state_np,
                    }
                    # print(f"[DEBUG] obs.wrist_image: shape={obs['wrist_image'].shape}, dtype={obs['wrist_image'].dtype}")
                    # print("[DEBUG] 观测数据字典内容:")
                    # for key, value in obs.items():
                    #     if isinstance(value, np.ndarray):
                    #         print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    #     else:
                    #         print(f"  {key}: type={type(value)}")

                    packed_obs = packer.pack(obs)
                    await ws.send(packed_obs)
                    # print("[DEBUG] 初始观测数据发送成功")
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[ERROR] WebSocket 在发送初始观测数据时连接关闭: {e}")
                    # break
                except Exception as e:
                    print(f"[ERROR] 发送初始观测数据失败: {e}")
                    # break

                while  not done and t < max_steps:
                    try:
                        action_bin = await ws.recv()
                        # print(f"[DEBUG] 收到动作消息: 长度={len(action_bin)} 字节")
                        action_msg = msgpack_numpy.unpackb(action_bin, raw=False)
                        
                        # 强制将其中的 numpy 数组变为可写副本
                        if isinstance(action_msg, dict) and "action" in action_msg:
                            action_msg["action"] = np.array(action_msg["action"]).copy()

                        action_msg = action_msg.copy()

                        # print(f"[DEBUG] 解包后的动作消息: {action_msg}")
                        if not isinstance(action_msg, dict) or "action" not in action_msg:
                            print("[ERROR] 动作消息格式错误: 期望包含 'action' 键的字典")
                            break
                        action = np.asarray(action_msg["action"])
                        if action.shape != (batch_size, 8) or action.dtype != np.float32:
                            print(f"[ERROR] 动作无效: 形状 {action.shape}, 类型 {action.dtype}")
                            break
                        action = action[0]
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"[ERROR] WebSocket 在接收动作时连接关闭: {e}")
                        # break
                    except Exception as e:
                        print(f"[ERROR] 接收或解析动作失败: {e}")
                        # break
                    action[7] = 1.0 if action[7] > 0.95 else 0.0
                    print(action)
                    # 四元数归一化
                    quaternion_raw = action[3:7]
                    norm = np.linalg.norm(quaternion_raw)
                    if norm > 0:
                        quaternion_normalized = quaternion_raw / norm
                    else:
                        quaternion_normalized = np.array([1.0, 0.0, 0.0, 0.0])
                    action[3:7] = quaternion_normalized

                    # 提取 gripper 控制值
                    gripper = action[7]

                    # 相对动作到绝对动作转换（前 7 维: xyz + 四元数）
                    action_world = compute_pose_in_world(arm_pose, action[:7])

                    # 拼接 gripper 值，构成完整的 8 维动作
                    action = np.concatenate([action_world, [gripper]])
                    
                    reward = 0
                    done = False
                    success = 0
                    try:
                        ts_obs, reward, done = env.step(np.concatenate([action[:7], [action[7]]]))
                        # print(f"[DEBUG] 环境步进: 奖励={reward}, 完成={done}")
                        # print(f"[DEBUG] Step {t + 1}: ts_obs.wrist_rgb mean={ts_obs.wrist_rgb.mean()}, gripper_pose={ts_obs.gripper_pose}, reward={reward}, done={done}")
                        
                        if not isinstance(reward, (int, float, np.floating)):
                            print(f"[ERROR] 奖励类型无效: 期望标量，实际 {type(reward)}")
                            
                        if not isinstance(done, (bool, np.bool_)):
                            print(f"[ERROR] 完成状态类型无效: 期望布尔值，实际 {type(done)}")
                            
                    except Exception as e:
                        print(f"[ERROR] 环境步进失败: {e}")
                        # break
                    if t == max_steps - 1:
                        done = True
                    
                    success = reward == 1.0 and np.isclose(action[7], 1.0)
                    feedback = {
                        "observation": {
                            "task": task_name,
                            "wrist_image": ts_obs.wrist_rgb[np.newaxis].copy(),
                            "front_image": ts_obs.front_rgb[np.newaxis].copy(),
                            "wrist_point_cloud": ts_obs.wrist_point_cloud.astype(np.float32)[np.newaxis].copy(),
                            "front_point_cloud": ts_obs.front_point_cloud.astype(np.float32)[np.newaxis].copy(),
                            "state": np.append(ts_obs.gripper_pose, ts_obs.gripper_open).astype(np.float32)[np.newaxis].copy(),
                        },
                        "reward": float(reward),
                        "terminated": bool(done),
                        "success": bool(success),
                    }
                    # 打印 feedback 中的 state
                    # print(f"[DEBUG] feedback['observation']['state']: value={feedback['observation']['state']}, "
                    #     f"shape={feedback['observation']['state'].shape}, dtype={feedback['observation']['state'].dtype}")
                    # print(feedback)
                    # print("-------------------")
                    # print(obs["wrist_image"] == feedback["observation"]["wrist_image"] )
                    # print(f"[DEBUG] feedback['observation']['wrist_image']: type={type(feedback['observation']['wrist_image'])}, shape={feedback['observation']['wrist_image'].shape}")
                    try:
                        transition = (feedback["observation"], feedback["reward"], feedback["terminated"])
                        packed_feedback = packer.pack(transition)
                        await ws.send(packed_feedback)
                        print("[DEBUG] 反馈数据发送成功")
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"[ERROR] WebSocket 在发送反馈数据时连接关闭: {e}")
                    except Exception as e:
                        print(f"[ERROR] 发送反馈数据失败: {e}")

                    rewards.append(reward)
                    print(f"Step {t + 1} - 奖励: {reward}, 成功: {success}")
                    if success:
                        done = True
                    t += 1
                    print(t)

                episode_return = np.sum(rewards)
                episode_returns.append(episode_return)
                highest_rewards.append(np.max(rewards) if rewards else 0.0)

            success_rate = np.mean(np.array(highest_rewards) == 1.0)
            avg_return = np.mean(episode_returns)
            print(f"\n成功率: {success_rate * 100:.2f}%")
            print(f"平均回报: {avg_return:.2f}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[ERROR] WebSocket 意外关闭: {e}")
    except Exception as e:
        print(f"[ERROR] 意外错误: {e}")
    finally:
        pass

if __name__ == "__main__":
    asyncio.run(run_client())