from __future__ import annotations

# global
import numpy as np
from rigmopy import utils_math as rp_math
from rigmopy import Quaternion, Pose, Vector3d

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


class EnvironmentPluggerPositionCtrl(Environment):
    """ Cartesian environment with position controller - Task: Peg in Hole """

    def __init__(self, **kwargs: dict[str, Any]):
        # Update environment configuration
        config_env = ch.search(kwargs, 'environment')
        Environment.__init__(self, config_env)
        # Extract component hyperparameter from kwargs
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_socket = ch.search(kwargs, 'socket')
        config_reward = ch.search(kwargs, 'reward')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_ft_sensor = ch.search(kwargs, 'ft_sensor')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_socket_sensor = ch.search(kwargs, 'socket_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Placeholder target sensor state
        self.x_arm2socket = Vector3d()
        self.q_arm2socket = Quaternion()
        # Components
        self.world = WorldPlugger(config_world, config_ur_arm, config_socket)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world.ur_arm)
        self.ft_sensor = FTSensor(config_ft_sensor, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.socket_sensor = SocketSensor(config_socket_sensor, self.world.ur_arm, self.world.socket)
        self.low_level_control = TcpPositionController(
            config_low_level_control,
            self.world.ur_arm,
            self.ik_solver,
            self.control_interface
        )
        self.reward = PoseWrenchReward(config_reward, self.clock)
        # self.reward = DistanceReward(config_reward, self.clock)

    def reset(self) -> npt.NDArray[np.float32]:
        # Reset environment
        self.clock.reset()
        if self.toggle_render_mode:
            self.world.disconnect()
            self.toggle_render_mode = False
        # Reset robot by default joint configuration
        self.world.reset(render=self.is_render)
        # Get start joint configuration by inverse kinematic
        X0_socket2plug = self.cfg.start_config.random(*self.cfg.reset_variance)
        X0_world2plug = self.world.socket.get_X_world2socket() * X0_socket2plug
        joint_pos0 = self.ik_solver.solve(X0_world2plug)
        # Reset robot again
        self.world.reset(joint_pos0)
        # Set new target pose
        self.x_arm2socket = self.socket_sensor.noisy_p_arm2sensor
        self.q_arm2socket = self.socket_sensor.noisy_q_arm2sensor
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], float, bool, dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # Apply action
        self.low_level_control.update(action=np.array(action))
        # Step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        self.ft_sensor.render_ft_bar(render=self.is_render)
        # Get new observation
        obs = self.get_obs()
        # Evaluate environment
        done = self.done
        X_arm2plug = self.plug_sensor.X_arm2sensor
        X_arm2socket = self.socket_sensor.X_arm2sensor
        F_tcp = self.ft_sensor.wrench
        reward = self.reward.compute(X_arm2plug, X_arm2socket, F_tcp, done)
        # reward = self.reward.compute(X_tcp, X_tgt, done)
        info = self.compose_info()
        return obs, reward, done, info

    def close(self) -> None:
        self.world.disconnect()

    def get_obs(self) -> npt.NDArray[np.float32]:
        # Build observation
        x_arm2plug = self.plug_sensor.p_arm2sensor
        q_arm2plug = self.plug_sensor.q_arm2sensor
        x_plug2socket = (self.x_arm2socket - x_arm2plug).xyz
        q_plug2socket = rp_math.quaternion_difference(q_arm2plug, self.q_arm2socket).wxyz
        F_tcp_meas = self.ft_sensor.noisy_wrench.xyzXYZ
        obs = np.array((x_plug2socket + q_plug2socket + F_tcp_meas), dtype=np.float32)
        # Evaluate metrics
        q_arm2socket_ = np.array(self.q_arm2socket.wxyz)
        q_arm2plug_ = np.array(q_arm2plug.wxyz)
        self.task_pos_error = np.sqrt(np.sum(np.square(x_plug2socket)))
        self.task_ang_error = np.arccos(np.clip((2 * (q_arm2socket_.dot(q_arm2plug_))**2 - 1), -1.0, 1.0))
        return obs

    def compose_info(self) -> dict[str, Any]:
        info = {
            'error_pos': self.task_pos_error,
            'error_ang': self.task_ang_error,
            'solved': self.solved,
        }
        return info
