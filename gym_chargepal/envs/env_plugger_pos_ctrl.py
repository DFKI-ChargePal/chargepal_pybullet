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

# mypy
from numpy import typing as npt
from typing import Any, Dict, Tuple


class EnvironmentPluggerPositionCtrl(Environment):
    """ Cartesian environment with position controller - Task: Peg in Hole """

    def __init__(self, **kwargs: Dict[str, Any]):
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
        # Start configuration in world coordinates
        self.x0_PW = self.cfg.target_config.p + self.cfg.start_config.p
        self.q0_PW = self.cfg.target_config.q * self.cfg.start_config.q
        self.X0_PW = Pose().from_pq(self.x0_PW, self.q0_PW)
        # Target sensor state
        self.x_SW = Vector3d()
        self.q_SW = Quaternion()
        # Resolve cross references
        config_world['ur_arm'] = config_ur_arm
        config_world['socket'] = config_socket
        config_low_level_control['plug_lin_config'] = self.x0_PW.xyz
        config_low_level_control['plug_ang_config'] = self.q0_PW.to_euler_angle()
        # Components
        self.world = WorldPlugger(config_world)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world.ur_arm)
        self.ft_sensor = FTSensor(config_ft_sensor, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.socket_sensor = SocketSensor(config_socket_sensor, self.world.socket)
        self.low_level_control = TcpPositionController(
            config_low_level_control,
            self.ik_solver,
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
        # Get start joint configuration by inverse kinematic
        X0 = self.X0_PW.random(*self.cfg.reset_variance)
        joint_pos0 = self.ik_solver.solve(X0)
        # Reset robot again
        self.world.reset(joint_pos0)
        # Set new target pose
        self.x_SW = self.socket_sensor.meas_pos()
        self.q_SW = self.socket_sensor.meas_ori()
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
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
        X_PW = Pose().from_pq(self.plug_sensor.get_pos(), self.plug_sensor.get_ori())
        X_SW = Pose().from_pq(self.socket_sensor.get_pos(), self.socket_sensor.get_ori())
        F_tcp = self.ft_sensor.get_wrench()
        reward = self.reward.compute(X_PW, X_SW, F_tcp, done)
        # reward = self.reward.compute(X_tcp, X_tgt, done)
        info = self.compose_info()
        return obs, reward, done, info

    def close(self) -> None:
        self.world.disconnect()

    def get_obs(self) -> npt.NDArray[np.float32]:
        # Build observation
        x_PW = self.plug_sensor.get_pos()
        q_PW = self.plug_sensor.get_ori()
        x_SP = (self.x_SW - x_PW).xyz
        q_SP = rp_math.quaternion_difference(q_PW, self.q_SW).wxyz
        F_tcp_meas = self.ft_sensor.meas_wrench().xyzXYZ
        obs = np.array((x_SP + q_SP + F_tcp_meas), dtype=np.float32)
        # Evaluate metrics
        q_SW_ = np.array(self.q_SW.wxyz)
        q_PW_ = np.array(q_PW.wxyz)
        self.task_pos_error = np.sqrt(np.sum(np.square(x_SP)))
        self.task_ang_error = np.arccos(np.clip((2 * (q_SW_.dot(q_PW_))**2 - 1), -1.0, 1.0))
        return obs

    def compose_info(self) -> Dict[str, Any]:
        info = {
            'error_pos': self.task_pos_error,
            'error_ang': self.task_ang_error,
            'solved': self.solved,
        }
        return info
