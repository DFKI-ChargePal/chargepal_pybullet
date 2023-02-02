# global
import numpy as np
import pybullet as p
import quaternionic as quat
from rigmopy import Orientation, Pose, Position

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.envs.env_base import Environment
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.bullet.ik_solver import IKSolver
# from gym_chargepal.utility.tf import Quaternion, Pose
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.worlds.world_reacher import WorldReacher
from gym_chargepal.sensors.sensor_joints import JointSensor
from gym_chargepal.sensors.sensor_virt_tgt import VirtTgtSensor
from gym_chargepal.reward.reward_dist_speed import DistanceSpeedReward
from gym_chargepal.controllers.controller_tcp_vel import TcpVelocityController
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

# mypy
from numpy import typing as npt
from typing import Any, Dict, Tuple


class EnvironmentReacherVelocityCtrl(Environment):
    """ Cartesian Environment with velocity controller - Task: point to point """
    def __init__(self, **kwargs: Dict[str, Any]):
        # Update environment configuration
        config_env = ch.search(kwargs, 'environment')
        Environment.__init__(self, config_env)
        # Extract component hyperparameter from kwargs
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_reward = ch.search(kwargs, 'reward')
        config_jacobian = ch.search(kwargs, 'jacobian')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_joint_sensor = ch.search(kwargs, 'joint_sensor')
        config_target_sensor = ch.search(kwargs, 'target_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Start configuration in world coordinates
        self.x0_WP = self.cfg.target_config.pos + self.cfg.start_config.pos
        self.q0_WP = self.cfg.target_config.ori * self.cfg.start_config.ori
        self.X0_WP = Pose(self.x0_WP, self.q0_WP)
        # Resolve cross references
        config_world['ur_arm'] = config_ur_arm
        config_world['target_config'] = self.cfg.target_config
        config_low_level_control['plug_lin_config'] = self.x0_WP.xyz
        config_low_level_control['plug_ang_config'] = p.getEulerFromQuaternion(self.q0_WP.xyzw)
        # Components
        self.world = WorldReacher(config_world)
        self.jacobian = Jacobian(config_jacobian, self.world.ur_arm)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface = JointVelocityMotorControl(config_control_interface, self.world.ur_arm)
        self.joint_sensor = JointSensor(config_joint_sensor, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.target_sensor = VirtTgtSensor(config_target_sensor, self.world)
        self.low_level_control = TcpVelocityController(
            config_low_level_control,
            self.jacobian, 
            self.control_interface,
            self.plug_sensor, 
            self.joint_sensor
        )
        self.reward = DistanceSpeedReward(config_reward, self.clock)

    def reset(self) -> npt.NDArray[np.float32]:
        # Reset environment
        self.clock.reset()
        if self.toggle_render_mode:
            self.world.disconnect()
            self.toggle_render_mode = False
        # Reset robot by default joint configuration
        self.world.reset(render=self.is_render)
        # Get start joint configuration by inverse kinematic
        X0 = self.X0_WP.random(*self.cfg.reset_variance)
        joint_config_0 = self.ik_solver.solve(X0.xyz_xyzw)
        # Reset robot again
        self.world.reset(joint_config_0)
        # Update sensors states
        self.update_sensors(target_sensor=True)
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # Apply action
        self.low_level_control.update(action=np.array(action))
        # Step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        # Update states
        self.update_sensors()
        obs = self.get_obs()
        # Evaluate environment
        done = self.done
        X_tcp = Pose(self.plug_sensor.get_pos(), self.plug_sensor.get_ori())
        X_tgt = Pose(self.plug_sensor.get_pos(), self.plug_sensor.get_ori())
        V_tcp = self.plug_sensor.get_twist()
        reward = self.reward.compute(X_tcp=X_tcp, V_tcp=V_tcp, X_tgt=X_tgt, done=done)
        info = self.compose_info()
        return obs, reward, done, info

    def close(self) -> None:
        self.world.disconnect()

    def update_sensors(self, target_sensor: bool=False) -> None:
        if target_sensor:
            self.target_sensor.update()

    def get_obs(self) -> npt.NDArray[np.float32]:
        # Get spatial distance
        dif_pos = (self.target_sensor.get_pos() - self.plug_sensor.get_pos()).xyz
        tgt_ori = self.target_sensor.get_ori().xyzw
        plg_ori = self.plug_sensor.get_ori().xyzw
        dif_ori = self.world.bullet_client.getDifferenceQuaternion(plg_ori, tgt_ori)
        # Get velocity signal
        twist = self.plug_sensor.get_twist().vw
        # Build observation
        obs = np.array((dif_pos + dif_ori + twist), dtype=np.float32)
        # Calculate side information
        tgt_ori_ = np.array(tgt_ori)
        plg_ori_ = np.array(plg_ori)
        self.task_pos_error = np.sqrt(np.sum(np.square(dif_pos)))
        self.task_ang_error = np.arccos(np.clip((2 * (tgt_ori_.dot(plg_ori_))**2 - 1), -1.0, 1.0))
        return obs

    def compose_info(self) -> Dict[str, Any]:
        info = {
            'error_pos': self.task_pos_error,
            'error_ang': self.task_ang_error,
            'solved': self.solved,
        }
        return info
