from __future__ import annotations

# global
import numpy as np
from rigmopy import utils_math as rp_math
from rigmopy import Quaternion, Vector3d

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.envs.env_base import Environment
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.sensors.sensor_ft import FTSensor
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.worlds.world_plugger import WorldPlugger
# from gym_chargepal.reward.reward_dist import DistanceReward
from gym_chargepal.sensors.sensor_socket import SocketSensor
from gym_chargepal.reward.reward_pose_wrench import PoseWrenchReward
from gym_chargepal.controllers.controller_tcp_pos import TcpPositionController
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl

# typing
from typing import Any
from numpy import typing as npt

ObsType = npt.NDArray[np.float32]
ActType = npt.NDArray[np.float32]


class EnvironmentPluggerPositionCtrl(Environment[ObsType, ActType]):
    """ Cartesian environment with position controller - Task: Peg in Hole """

    def __init__(self, **kwargs: dict[str, Any]):
        # Update environment configuration
        config_env = ch.search(kwargs, 'environment')
        Environment.__init__(self, config_env)
        # Extract component hyperparameter from kwargs
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_start = ch.search(kwargs, 'start')
        config_socket = ch.search(kwargs, 'socket')
        config_reward = ch.search(kwargs, 'reward')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_ft_sensor = ch.search(kwargs, 'ft_sensor')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_socket_sensor = ch.search(kwargs, 'socket_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Placeholder noisy target sensor state
        self.noisy_p_arm2socket = Vector3d()
        self.noisy_q_arm2socket = Quaternion()
        # Manipulate default configuration
        if config_ur_arm.get('tcp_link_name') is None:
            config_ur_arm['tcp_link_name'] = 'plug_lip'
        # Components
        self.world: WorldPlugger = WorldPlugger(config_world, config_ur_arm, config_start, config_socket)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world.ur_arm)
        self.ft_sensor = FTSensor(config_ft_sensor, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.socket_sensor = SocketSensor(config_socket_sensor, self.world.ur_arm, self.world.socket)
        self.controller = TcpPositionController(
            config_low_level_control,
            self.world.ur_arm,
            self.ik_solver,
            self.control_interface
        )
        self.reward = PoseWrenchReward(config_reward, self.clock)
        # self.reward = DistanceReward(config_reward, self.clock)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        # Reset environment
        obs, info = self._reset_core()
        # Set new target pose
        self.noisy_p_arm2socket = self.socket_sensor.noisy_p_arm2sensor
        self.noisy_q_arm2socket = self.socket_sensor.noisy_q_arm2sensor
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # Perform core step
        obs, terminated, truncated, info = self._update_core(action)
        reward = self.reward.compute(
            self.world.ur_arm.X_arm2plug, self.world.socket.X_arm2socket, self.world.ur_arm.wrench, terminated)
        return obs, reward, terminated, truncated, info

    def get_obs(self) -> npt.NDArray[np.float32]:
        # Build noisy observation
        noisy_p_plug2socket = (self.noisy_p_arm2socket - self.plug_sensor.noisy_p_arm2sensor).xyz
        noisy_q_plug2socket = rp_math.quaternion_difference(self.plug_sensor.noisy_q_arm2sensor, self.noisy_q_arm2socket).wxyz
        noisy_F_plug = self.ft_sensor.noisy_wrench.xyzXYZ
        # Glue observation together
        obs = np.array((noisy_p_plug2socket + noisy_q_plug2socket + noisy_F_plug), dtype=np.float32)
        return obs

    def compose_info(self) -> dict[str, Any]:
        # Calculate evaluation metrics
        p_plug2target = (self.world.socket.p_arm2socket - self.world.ur_arm.p_arm2plug).xyz
        q_arm2tgt = np.array(self.world.socket.q_arm2socket.wxyz)
        q_arm2plug = np.array(self.world.ur_arm.q_arm2plug.wxyz)
        self.task_pos_error = np.sqrt(np.sum(np.square(p_plug2target)))
        self.task_ang_error = np.arccos(np.clip((2 * (q_arm2tgt.dot(q_arm2plug))**2 - 1), -1.0, 1.0))
        info = {
            'error_pos': self.task_pos_error,
            'error_ang': self.task_ang_error,
            'solved': self.solved,
        }
        return info
