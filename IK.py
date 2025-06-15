import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

# 启动仿真环境
physicsClient = p.connect(p.DIRECT)  # 或 p.GUI 来可视化

# 加载UR5机器人模型（假设UR5的URDF文件路径为"ur5.urdf"）
robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# 目标位置 (x, y, z)
target_position = [0.5, 0.5, 0.5]

# 目标姿态 (四元数或欧拉角)
# 假设我们传递欧拉角作为目标姿态 (roll, pitch, yaw)
euler_angles = [0.0, np.pi/2, 0.0]  # 这里是90度的旋转，可以修改为实际的目标姿态

# 将欧拉角转换为四元数
r = R.from_euler('xyz', euler_angles, degrees=False)
target_quaternion = r.as_quat()  # [rx, ry, rz, rw]

# 计算逆运动学
# 假设我们要求解的是末端执行器位置和姿态的逆解
joint_angles = p.calculateInverseKinematics(
    robot_id,  # 机器人ID
    endEffectorLinkIndex=11,  # UR5的末端执行器链接索引
    targetPosition=target_position,  # 目标位置 (x, y, z)
    targetOrientation=target_quaternion,  # 目标姿态 (四元数)
    lowerLimits=None,  # 可以根据需要定义关节的下限
    upperLimits=None,  # 可以根据需要定义关节的上限
    jointRanges=None,  # 关节的范围
    restPoses=None  # 末端执行器的初始位置
)

# 输出求得的关节角度
print("Inverse Kinematics Joint Angles:", joint_angles)

# 断开仿真环境
p.disconnect()
