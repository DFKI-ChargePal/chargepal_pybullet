from __future__ import annotations

# global
import numpy as np
from rigmopy import utils_math as rp_math
from rigmopy import Pose

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.envs.env_base import Environment
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.reward.reward_dist import DistanceReward
from gym_chargepal.worlds.world_reacher import WorldReacher
from gym_chargepal.sensors.sensor_virt_tgt import VirtTgtSensor
from gym_chargepal.controllers.controller_tcp_pos import TcpPositionController
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl

# typing
from typing import Any
from numpy import typing as npt


class EnvironmentReacherPositionCtrl(Environment):
    """ Cartesian Environment with position controller - Task: point to point """
    def __init__(self, **kwargs: dict[str, Any]):
        # Update environment configuration
        config_env = ch.search(kwargs, 'environment')
        Environment.__init__(self, config_env)
        # Extract component hyperparameter from kwargs
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_reward = ch.search(kwargs, 'reward')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_target_sensor = ch.search(kwargs, 'target_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Start configuration in world coordinates
        self.p0_arm2plug = self.cfg.target_config.p + self.cfg.start_config.p
        self.q0_arm2plug = self.cfg.target_config.q * self.cfg.start_config.q
        # self.X0_arm2plug = Pose().from_pq(self.p0_arm2plug, self.q0_arm2plug)
        self.X0_arm2plug = self.cfg.target_config * self.cfg.start_config
        # Resolve cross references
        config_world['ur_arm'] = config_ur_arm
        config_world['target_config'] = self.cfg.target_config
        # Components
        self.world = WorldReacher(config_world)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.target_sensor = VirtTgtSensor(config_target_sensor, self.world)
        self.low_level_control = TcpPositionController(
            config_low_level_control,
            self.world.ur_arm,
            self.ik_solver,
            self.control_interface
        )
        self.reward = DistanceReward(config_reward, self.clock)

    def reset(self) -> npt.NDArray[np.float32]:
        # Reset environment
        self.clock.reset()
        if self.toggle_render_mode:
            self.world.disconnect()
            self.toggle_render_mode = False
        # Reset robot by default joint configuration
        self.world.reset(render=self.is_render)
        # Get start joint configuration by inverse kinematic
        X_world2arm = self.world.ur_arm.base_link.get_X_world2link()
        X0_world2plug = X_world2arm * self.X0_arm2plug
        X0 = X0_world2plug.random(*self.cfg.reset_variance)
        joint_config_0 = self.ik_solver.solve(X0)
        # Reset robot again
        self.world.reset(joint_config_0)
        self.low_level_control.reset()
        # Update sensors states
        self.update_sensors(target_sensor=True)
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], float, bool, dict[Any, Any]]:
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
        X_PW = Pose().from_pq(self.plug_sensor.p_arm2sensor, self.plug_sensor.q_arm2sensor)
        X_SW = Pose().from_pq(self.target_sensor.get_pos(), self.target_sensor.get_ori())
        reward = self.reward.compute(X_PW, X_SW, done)
        info = self.compose_info()
        return obs, reward, done, info

    def close(self) -> None:
        self.world.disconnect()

    def update_sensors(self, target_sensor: bool=False) -> None:
        if target_sensor:
            self.target_sensor.update()

    def get_obs(self) -> npt.NDArray[np.float32]:
        # Build observation
        p_arm2plug_arm = self.plug_sensor.p_arm2sensor
        p_arm2tgt_arm = self.target_sensor.get_pos()
        # Translation plug to target in arm frame
        p_plug2target_arm = (p_arm2tgt_arm - p_arm2plug_arm).xyz

        q_arm2plug = self.plug_sensor.q_arm2sensor
        q_arm2tgt = self.target_sensor.get_ori()
        # Minimal rotation plug to target
        q_plug2tgt = rp_math.quaternion_difference(q_arm2plug, q_arm2tgt).wxyz
        # Glue observation together
        obs = np.array((p_plug2target_arm + q_plug2tgt), dtype=np.float32)
        # Calculate evaluation metrics
        q_arm2tgt_ = np.array(q_arm2tgt.wxyz)
        q_arm2plug_ = np.array(q_arm2plug.wxyz)
        self.task_pos_error = np.sqrt(np.sum(np.square(p_plug2target_arm)))
        self.task_ang_error = np.arccos(np.clip((2 * (q_arm2tgt_.dot(q_arm2plug_))**2 - 1), -1.0, 1.0))
        return obs

    def compose_info(self) -> dict[str, Any]:
        info = {
            'error_pos': self.task_pos_error,
            'error_ang': self.task_ang_error,
            'solved': self.solved,
        }
        return info
