from pyrep.backend import sim, utils
from pyrep.objects import Object
from pyrep.objects.dummy import Dummy
from pyrep.robots.configuration_paths.arm_configuration_path import (
    ArmConfigurationPath)
from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.errors import ConfigurationError, ConfigurationPathError, IKError
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.const import PYREP_SCRIPT_TYPE
from typing import List, Union
import numpy as np
import warnings


class Arm(RobotComponent):
    """Base class representing a robot arm with path planning support.
    """

    def __init__(self, count: int, name: str, num_joints: int,
                 base_name: str = None,
                 max_velocity=1.0, max_acceleration=4.0, max_jerk=1000):
        """Count is used for when we have multiple copies of arms"""
        joint_names = ['%s_joint%d' % (name, i+1) for i in range(num_joints)]
        super().__init__(count, name, joint_names, base_name)
        self.name = name 
        # Used for motion planning
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        
        # Motion planning handles
        suffix = '' if count == 0 else '#%d' % (count - 1)
        self._ik_target = Dummy('%s_target%s' % (name, suffix))
        self._ik_tip = Dummy('%s_tip%s' % (name, suffix))
        self._ik_group = sim.simGetIkGroupHandle('%s_ik%s' % (name, suffix))
        self._collision_collection = sim.simGetCollectionHandle(
            '%s_arm%s' % (name, suffix))

    def set_ik_element_properties(self, constraint_x=True, constraint_y=True,
                                  constraint_z=True,
                                  constraint_alpha_beta=True,
                                  constraint_gamma=True) -> None:
        constraints = 0
        if constraint_x:
            constraints |= sim.sim_ik_x_constraint
        if constraint_y:
            constraints |= sim.sim_ik_y_constraint
        if constraint_z:
            constraints |= sim.sim_ik_z_constraint
        if constraint_alpha_beta:
            constraints |= sim.sim_ik_alpha_beta_constraint
        if constraint_gamma:
            constraints |= sim.sim_ik_gamma_constraint
        sim.simSetIkElementProperties(
            ikGroupHandle=self._ik_group,
            tipDummyHandle=self._ik_tip.get_handle(),
            constraints=constraints,
            precision=None,
            weight=None,
        )

    def set_ik_group_properties(self, resolution_method='pseudo_inverse', max_iterations=6, dls_damping=0.1) -> None:
        try:
            res_method = {'pseudo_inverse': sim.sim_ik_pseudo_inverse_method,
                          'damped_least_squares': sim.sim_ik_damped_least_squares_method,
                          'jacobian_transpose': sim.sim_ik_jacobian_transpose_method}[resolution_method]
        except KeyError:
            raise Exception('Invalid resolution method,'
                            'Must be one of ["pseudo_inverse" | "damped_least_squares" | "jacobian_transpose"]')
        sim.simSetIkGroupProperties(
            ikGroupHandle=self._ik_group,
            resolutionMethod=res_method,
            maxIterations=max_iterations,
            damping=dls_damping
        )

    def solve_ik_via_sampling(self,
                              position: Union[List[float], np.ndarray],
                              euler: Union[List[float], np.ndarray] = None,
                              quaternion: Union[List[float], np.ndarray] = None,
                              ignore_collisions: bool = False,
                              trials: int = 300,
                              max_configs: int = 1,
                              distance_threshold: float = 0.65,
                              max_time_ms: int = 10,
                              relative_to: Object = None
                              ) -> np.ndarray:
        """通过采样求解逆运动学并返回计算得到的关节值。

        该逆运动学方法通过随机搜索符合目标末端执行器位姿的机械臂配置。在目标末端位姿足够接近
        后，计算逆运动学以尽量将末端执行器带到目标位置。该方法适用于起始位姿与目标位姿之间的距离较大，
        线性化方法（如雅可比法）不再适用的情况。

        我们会在指定的尝试次数 `trials` 内生成最多 `max_configs` 个样本，并根据关节角度的距离对其进行排序。

        必须指定欧拉角或四元数中的一个，但不能同时指定两者！

        :param position: 目标位置的 x, y, z 坐标。
        :param euler: 目标的欧拉角（绕 x, y, z 轴的旋转，单位为弧度）。
        :param quaternion: 包含四元数（x, y, z, w）的列表。
        :param ignore_collisions: 是否禁用碰撞检测。
        :param trials: 最大尝试次数，用于生成最大数量的配置。
        :param max_configs: 需要生成的最大有效配置数，在达到此数目后排序。
        :param distance_threshold: 当末端执行器的距离接近目标时，开始进行逆运动学计算。
        :param max_time_ms: 每次配置搜索的最大时间，单位为毫秒。
        :param relative_to: 目标位姿的参考框架。若为 None，则表示使用绝对位姿；若指定为一个 `Object`，则表示相对于该对象的参考框架。
        :raises: ConfigurationError 如果未找到有效的关节配置。

        :return: 返回 `max_configs` 个关节配置，按角度距离排序。
        """
        # 检查用户是否正确指定了欧拉角或四元数，不能同时指定两者
        if not ((euler is None) ^ (quaternion is None)):
            raise ConfigurationError(
                '请指定欧拉角或四元数中的一个，但不能同时指定两者。')

        # 保存当前的末端执行器位姿
        prev_pose = self._ik_target.get_pose()

        # 设置目标位置
        self._ik_target.set_position(position, relative_to)

        # 设置目标姿态（通过欧拉角或四元数）
        if euler is not None:
            self._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            self._ik_target.set_quaternion(quaternion, relative_to)

        # 获取机械臂的关节句柄
        handles = [j.get_handle() for j in self.joints]

        # 获取关节的周期性和区间
        cyclics, intervals = self.get_joint_intervals()
        low_limits, max_limits = list(zip(*intervals))

        # 对关节的范围进行限制，避免过大
        low_limits = np.maximum(low_limits, -np.pi * 2).tolist()
        max_limits = np.minimum(max_limits, np.pi * 2).tolist()

        # 碰撞对列表，如果不忽略碰撞，设置默认的碰撞检测
        collision_pairs = []
        if not ignore_collisions:
            collision_pairs = [self._collision_collection, sim.sim_handle_all]

        # 初始化需要的变量
        metric = joint_options = None
        valid_joint_positions = []

        # 开始进行随机采样的尝试
        for i in range(trials):
            # 通过采样获取符合目标位姿的关节配置
            config = sim.simGetConfigForTipPose(
                self._ik_group, handles, distance_threshold, int(max_time_ms),
                metric, collision_pairs, joint_options, low_limits, max_limits)

            # 如果得到了有效配置，则将其加入有效配置列表
            if len(config) > 0:
                valid_joint_positions.append(config)

            # 如果已经达到了最大配置数，则停止采样
            if len(valid_joint_positions) >= max_configs:
                break

        # 恢复之前的末端执行器位姿
        self._ik_target.set_pose(prev_pose)

        # 如果没有找到任何有效配置，抛出异常
        if len(valid_joint_positions) == 0:
            raise ConfigurationError(
                '未能找到满足目标末端执行器位姿的有效关节配置。')

        # 如果找到了多个有效配置，则根据与当前配置的角度距离进行排序
        if len(valid_joint_positions) > 1:
            current_config = np.array(self.get_joint_positions())
            # 根据关节配置之间的角度距离对有效配置进行排序
            valid_joint_positions.sort(
                key=lambda x: np.linalg.norm(current_config - x))

        # 返回排序后的关节配置列表
        return np.array(valid_joint_positions)

    def get_configs_for_tip_pose(self,
                                 position: Union[List[float], np.ndarray],
                                 euler: Union[List[float], np.ndarray] = None,
                                 quaternion: Union[List[float], np.ndarray] = None,
                                 ignore_collisions=False,
                                 trials=300, max_configs=60,
                                 relative_to: Object = None
                                 ) -> List[List[float]]:
        """Gets a valid joint configuration for a desired end effector pose.
        Must specify either rotation in euler or quaternions, but not both!
        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs
        :param max_configs: The maximum number of configurations we want to
            generate before ranking them.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationError if no joint configuration could be found.
        :return: A list of valid joint configurations for the desired
        end effector pose.
        """

        warnings.warn("Please use 'solve_ik_via_sampling' instead.",
                      DeprecationWarning)
        return list(self.solve_ik_via_sampling(
            position, euler, quaternion, ignore_collisions, trials,
            max_configs, relative_to=relative_to))

    def solve_ik_via_jacobian(
            self, position: Union[List[float], np.ndarray],
            euler: Union[List[float], np.ndarray] = None,
            quaternion: Union[List[float], np.ndarray] = None,
            relative_to: Object = None) -> List[float]:
        """通过雅可比矩阵求解逆运动学，并返回计算得到的关节值。

        该逆运动学方法通过当前机器人配置进行线性化。此线性化假设起始和目标位姿之间的距离较近，
        在这种情况下有效。但如果目标与起始位姿的距离过大，线性化方法将不再适用，用户最好使用
        'solve_ik_via_sampling' 方法进行求解。

        必须指定旋转参数（欧拉角或四元数），但不能同时指定两者！

        :param position: 目标位置的 x, y, z 坐标。
        :param euler: 目标的欧拉角（绕 x, y, z 轴的旋转，单位为弧度）。
        :param quaternion: 包含四元数（x, y, z, w）的列表。
        :param relative_to: 指定目标位姿是相对于哪个参考框架的。若为 None，则表示绝对位姿；若为
                            一个 `Object`，则表示相对于该对象的参考框架计算目标位姿。
        :return: 返回一个包含计算得到的关节值的列表。

        详细流程：
        1. 将目标位置 `position` 设置到目标对象 `_ik_target`，如果指定了相对参考框架（`relative_to`），
           则目标位姿是相对于该参考框架的。
        2. 根据输入的旋转参数 `euler` 或 `quaternion`，设置目标的姿态（方向）。确保 `euler` 和
           `quaternion` 至少提供其中之一，但不能同时提供。
        3. 调用 `sim.simCheckIkGroup` 方法进行逆运动学计算，检查是否能够求解目标位姿下的关节值。
           该方法返回一个结果标志和关节值。
        4. 如果求解失败（`sim_ikresult_fail`），抛出 `IKError` 异常，提示用户目标与末端执行器之间的距离
           可能过大，导致线性化方法不再有效。
        5. 如果逆运动学没有执行（`sim_ikresult_not_performed`），抛出 `IKError` 异常，说明没有进行求解。
        6. 如果求解成功，则返回计算得到的关节值。

        示例：
            position = [0.5, 0.2, 0.3]   # 目标位置
            euler = [0.0, 0.0, 1.57]     # 目标欧拉角
            quaternion = [0.0, 0.0, 0.707, 0.707]  # 目标四元数（不需要同时传递）
            joint_values = self.solve_ik_via_jacobian(position, euler=euler)
        """

        # 设置目标位置
        self._ik_target.set_position(position, relative_to)

        # 设置目标方向（通过欧拉角或四元数）
        if euler is not None:
            self._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            self._ik_target.set_quaternion(quaternion, relative_to)

        # 调用逆运动学求解函数
        ik_result, joint_values = sim.simCheckIkGroup(
            self._ik_group, [j.get_handle() for j in self.joints]
        )

        # 判断逆运动学求解结果
        if ik_result == sim.sim_ikresult_fail:
            # 如果求解失败，抛出异常并给出提示信息
            raise IKError('IK failed. Perhaps the distance between the tip '
                          'and the target was too large.')
        elif ik_result == sim.sim_ikresult_not_performed:
            # 如果没有执行逆运动学求解，抛出异常
            raise IKError('IK not performed.')

        # 返回计算得到的关节值
        return joint_values

    def solve_ik(self, position: Union[List[float], np.ndarray],
                 euler: Union[List[float], np.ndarray] = None,
                 quaternion: Union[List[float], np.ndarray] = None,
                 relative_to: Object = None) -> List[float]:
        """Solves an IK group and returns the calculated joint values.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :return: A list containing the calculated joint values.
        """
        warnings.warn("Please use 'solve_ik_via_jacobian' instead.",
                      DeprecationWarning)
        return self.solve_ik_via_jacobian(
            position, euler, quaternion, relative_to)

    def get_path_from_cartesian_path(self, path: CartesianPath
                                     ) -> ArmConfigurationPath:
        """Translate a path from cartesian space, to arm configuration space.

        Note: It must be possible to reach the start of the path via a linear
        path, otherwise an error will be raised.

        :param path: A :py:class:`CartesianPath` instance to be translated to
            a configuration-space path.
        :raises: ConfigurationPathError if no path could be created.

        :return: A path in the arm configuration space.
        """
        handles = [j.get_handle() for j in self.joints]
        _, ret_floats, _, _ = utils.script_call(
            'getPathFromCartesianPath@PyRep', PYREP_SCRIPT_TYPE,
            ints=[path.get_handle(), self._ik_group,
                  self._ik_target.get_handle()] + handles)
        if len(ret_floats) == 0:
            raise ConfigurationPathError(
                'Could not create a path from cartesian path.')
        return ArmConfigurationPath(self, ret_floats)

    def get_linear_path(self, position: Union[List[float], np.ndarray],
                        euler: Union[List[float], np.ndarray] = None,
                        quaternion: Union[List[float], np.ndarray] = None,
                        steps=50, ignore_collisions=False,
                        relative_to: Object = None) -> ArmConfigurationPath:
        """Gets a linear configuration path given a target pose.

        Generates a path that drives a robot from its current configuration
        to its target dummy in a straight line (i.e. shortest path in Cartesian
        space).

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param steps: The desired number of path points. Each path point
            contains a robot configuration. A minimum of two path points is
            required. If the target pose distance is large, a larger number
            of steps leads to better results for this function.
        :param ignore_collisions: If collision checking should be disabled.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationPathError if no path could be created.

        :return: A linear path in the arm configuration space.
        """
        if not ((euler is None) ^ (quaternion is None)):
            raise ConfigurationPathError(
                'Specify either euler or quaternion values, but not both.')

        prev_pose = self._ik_target.get_pose()
        self._ik_target.set_position(position, relative_to)
        if euler is not None:
            self._ik_target.set_orientation(euler, relative_to)
        elif quaternion is not None:
            self._ik_target.set_quaternion(quaternion, relative_to)
        handles = [j.get_handle() for j in self.joints]

        collision_pairs = []
        if not ignore_collisions:
            collision_pairs = [self._collision_collection, sim.sim_handle_all]
        joint_options = None
        ret_floats = sim.generateIkPath(
            self._ik_group, handles, steps, collision_pairs, joint_options)
        self._ik_target.set_pose(prev_pose)
        if len(ret_floats) == 0:
            # print("ConfigurationPathError")
            raise ConfigurationPathError('Could not create path.')
        return ArmConfigurationPath(self, ret_floats)

    def get_nonlinear_path(self, position: Union[List[float], np.ndarray],
                           euler: Union[List[float], np.ndarray] = None,
                           quaternion: Union[List[float], np.ndarray] = None,
                           ignore_collisions=False,
                           trials=300,
                           max_configs=1,
                           distance_threshold: float = 0.65,
                           max_time_ms: int = 10,
                           trials_per_goal=1,
                           algorithm=Algos.SBL,
                           relative_to: Object = None
                           ) -> ArmConfigurationPath:
        """Gets a non-linear (planned) configuration path given a target pose.

        A path is generated by finding several configs for a pose, and ranking
        them according to the distance in configuration space (smaller is
        better).

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs.
            See 'solve_ik_via_sampling'.
        :param max_configs: The maximum number of configurations we want to
            generate before sorting them. See 'solve_ik_via_sampling'.
        :param distance_threshold: Distance indicating when IK should be
            computed in order to try to bring the tip onto the target.
            See 'solve_ik_via_sampling'.
        :param max_time_ms: Maximum time in ms spend searching for
            each configuation. See 'solve_ik_via_sampling'.
        :param trials_per_goal: The number of paths per config we want to trial.
        :param algorithm: The algorithm for path planning to use.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.
        :raises: ConfigurationPathError if no path could be created.

        :return: A non-linear path in the arm configuration space.
        """

        handles = [j.get_handle() for j in self.joints]

        try:
            configs = self.solve_ik_via_sampling(
                position, euler, quaternion, ignore_collisions, trials,
                max_configs, distance_threshold, max_time_ms, relative_to)
        except ConfigurationError as e:
            raise ConfigurationPathError('Could not create path.') from e

        _, ret_floats, _, _ = utils.script_call(
            'getNonlinearPath@PyRep', PYREP_SCRIPT_TYPE,
            ints=[self._collision_collection, int(ignore_collisions),
                  trials_per_goal] + handles,
            floats=configs.flatten().tolist(),
            strings=[algorithm.value])

        if len(ret_floats) == 0:
            raise ConfigurationPathError('Could not create path.')
        return ArmConfigurationPath(self, ret_floats)

    def get_path(self, position: Union[List[float], np.ndarray],
                 euler: Union[List[float], np.ndarray] = None,
                 quaternion: Union[List[float], np.ndarray] = None,
                 ignore_collisions=False,
                 trials=300,
                 max_configs=1,
                 distance_threshold: float = 0.65,
                 max_time_ms: int = 10,
                 trials_per_goal=1,
                 algorithm=Algos.SBL,
                 relative_to: Object = None,
                 steps=50
                 ) -> ArmConfigurationPath:
        """Tries to get a linear path, failing that tries a non-linear path.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y, z position of the target.
        :param euler: The x, y, z orientation of the target (in radians).
        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param ignore_collisions: If collision checking should be disabled.
        :param trials: The maximum number of attempts to reach max_configs.
            See 'solve_ik_via_sampling'.
        :param max_configs: The maximum number of configurations we want to
            generate before sorting them. See 'solve_ik_via_sampling'.
        :param distance_threshold: Distance indicating when IK should be
            computed in order to try to bring the tip onto the target.
            See 'solve_ik_via_sampling'.
        :param max_time_ms: Maximum time in ms spend searching for
            each configuation. See 'solve_ik_via_sampling'.
        :param trials_per_goal: The number of paths per config we want to trial.
        :param algorithm: The algorithm for path planning to use.
        :param relative_to: Indicates relative to which reference frame we want
        the target pose. Specify None to retrieve the absolute pose,
        or an Object relative to whose reference frame we want the pose.

        :raises: ConfigurationPathError if neither a linear or non-linear path
            can be created.
        :return: A linear or non-linear path in the arm configuration space.
        """
        try:
            p = self.get_linear_path(position, euler, quaternion,
                                     ignore_collisions=ignore_collisions,
                                     relative_to=relative_to,
                                     steps=steps)
            return p, True

        except ConfigurationPathError:
            pass  # Allowed. Try again, but with non-linear.

        # This time if an exception is thrown, we dont want to catch it.
        p = self.get_nonlinear_path(
            position, euler, quaternion, ignore_collisions, trials, max_configs,
            distance_threshold, max_time_ms, trials_per_goal, algorithm,
            relative_to)
        return p, False

    def get_tip(self) -> Dummy:
        """Gets the tip of the arm.

        Each arm is required to have a tip for path planning.

        :return: The tip of the arm.
        """
        return self._ik_tip

    def get_jacobian(self):
        """Calculates the Jacobian.

        :return: the row-major Jacobian matix.
        """
        self._ik_target.set_matrix(self._ik_tip.get_matrix())
        sim.simCheckIkGroup(self._ik_group,
                            [j.get_handle() for j in self.joints])
        jacobian, (rows, cols) = sim.simGetIkGroupMatrix(self._ik_group, 0)
        jacobian = np.array(jacobian).reshape((rows, cols), order='F')
        return jacobian

    def check_arm_collision(self, obj: 'Object' = None) -> bool:
        """Checks whether two entities are colliding.

        :param obj: The other collidable object to check collision against,
            or None to check against all collidable objects. Note that objects
            must be marked as collidable!
        :return: If the object is colliding.
        """
        handle = sim.sim_handle_all if obj is None else obj.get_handle()
        return sim.simCheckCollision(self._collision_collection, handle) == 1
