from __future__ import annotations

# global
import numpy as np
from rigmopy import utils_math as rp_math
from rigmopy import Quaternion, Vector3d
from dataclasses import dataclass

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.envs.env_base import Environment, EnvironmentCfg
from gym_chargepal.sensors.sensor_ft import FTSensor
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.worlds.world_plugger import WorldPlugger
from gym_chargepal.sensors.sensor_socket import SocketSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
# from gym_chargepal.reward.reward_dist import DistanceReward
from gym_chargepal.reward.reward_pose_wrench import PoseWrenchReward
from gym_chargepal.controllers.controller_tcp_motion import TCPMotionController
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

# typing
from typing import Any
from numpy import typing as npt

@dataclass
class EnvironmentPluggerMotionCtrlCfg(EnvironmentCfg):
    hw_interface: str = 'joint_velocity'


class EnvironmentPluggerMotionCtrl(Environment):
    """ Cartesian environment with compliance controller - Task: Peg in Hole """

    def __init__(self, **kwargs: dict[str, Any]):
        # Update environment configuration
        config_env = ch.search(kwargs, 'environment')
        Environment.__init__(self, config_env)
        # create configuration and overwrite values
        self.cfg: EnvironmentPluggerMotionCtrlCfg = EnvironmentPluggerMotionCtrlCfg()
        self.cfg.update(**config_env)
        # Extract component hyperparameter from kwargs
        config_start = ch.search(kwargs, 'start')
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_socket = ch.search(kwargs, 'socket')
        config_reward = ch.search(kwargs, 'reward')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_ft_sensor = ch.search(kwargs, 'ft_sensor')
        config_virtual_arm = ch.search(kwargs, 'virtual_arm')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_socket_sensor = ch.search(kwargs, 'socket_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Placeholder noisy target sensor state
        self.noisy_p_arm2socket = Vector3d()
        self.noisy_q_arm2socket = Quaternion()
        # Components
        self.world = WorldPlugger(config_world, config_ur_arm, config_start, config_socket)
        config_virtual_arm['tcp_link_name'] = self.world.ur_arm.cfg.tcp_link_name
        self.virtual_arm = VirtualURArm(config_virtual_arm, self.world)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface: JointPositionMotorControl | JointVelocityMotorControl
        if self.cfg.hw_interface == 'joint_velocity':
            self.control_interface = JointVelocityMotorControl(config_control_interface, self.world.ur_arm)
        elif self.cfg.hw_interface == 'joint_position':
            self.control_interface = JointPositionMotorControl(config_control_interface, self.world.ur_arm)
        else:
            raise ValueError(f"Unknown Hardware interface '{self.cfg.hw_interface}'. "
                             f"Options are: 'joint_velocity', 'joint_position'")
        self.ft_sensor = FTSensor(config_ft_sensor, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.socket_sensor = SocketSensor(config_socket_sensor, self.world.ur_arm, self.world.socket)
        config_low_level_control['period'] = self.world.ctrl_period
        self.low_level_control = TCPMotionController(
            config_low_level_control,
            self.world.ur_arm,
            self.virtual_arm,
            self.control_interface,
            self.plug_sensor
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
        # Get start joint start configuration by inverse kinematic
        joint_pos0 = self.ik_solver.solve(self.world.sample_X0())
        self.world.reset(joint_conf=joint_pos0, render=self.is_render)
        self.low_level_control.reset()
        # Set new target pose
        self.noisy_p_arm2socket = self.socket_sensor.noisy_p_arm2sensor
        self.noisy_q_arm2socket = self.socket_sensor.noisy_q_arm2sensor
        return self.get_obs()
    
    def step(self, action: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], float, bool, dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # Apply action
        self.low_level_control.update(action=np.array(action))
        # Step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        self.ft_sensor.render_ft_bar(render=self.is_render)
        # Update and evaluate state
        done = self.done
        obs = self.get_obs()
        info = self.compose_info()
        # reward = self.reward.compute(self.world.ur_arm.X_arm2plug, self.world.socket.X_arm2socket, done)
        reward = self.reward.compute(
            self.world.ur_arm.X_arm2plug, self.world.socket.X_arm2socket, self.world.ur_arm.wrench, done)
        return obs, reward, done, info

    def close(self) -> None:
        self.world.disconnect()

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
