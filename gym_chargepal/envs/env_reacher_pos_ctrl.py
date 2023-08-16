from __future__ import annotations

# global
import numpy as np
from rigmopy import utils_math as rp_math
from rigmopy import Pose, Quaternion, Vector3d

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

ObsType = npt.NDArray[np.float32]
ActType = npt.NDArray[np.float32]


class EnvironmentReacherPositionCtrl(Environment[ObsType, ActType]):
    """ Cartesian Environment with position controller - Task: point to point """
    def __init__(self, **kwargs: Any):
        # Update environment configuration
        config_env = ch.search(kwargs, 'environment')
        Environment.__init__(self, config_env, render_mode=kwargs.get('render_mode'))
        # Extract component hyperparameter from kwargs
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_start = ch.search(kwargs, 'start')
        config_target = ch.search(kwargs, 'target')
        config_reward = ch.search(kwargs, 'reward')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_target_sensor = ch.search(kwargs, 'target_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Placeholder noisy target sensor state
        self.noisy_p_arm2tgt = Vector3d()
        self.noisy_q_arm2tgt = Quaternion()
        # Components
        self.world = WorldReacher(config_world, config_ur_arm, config_start, config_target)
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        # Reset environment
        self.clock.reset()
        # Reset robot by default joint configuration
        self.world.reset(render=self.is_render)
        # Get start joint start configuration by inverse kinematic
        joint_pos0 = self.ik_solver.solve(self.world.sample_X0())
        self.world.reset(joint_conf=joint_pos0, render=self.is_render)
        self.low_level_control.reset()
        # Set new noisy target pose
        self.noisy_p_arm2tgt = self.target_sensor.noisy_p_arm2tgt
        self.noisy_q_arm2tgt = self.target_sensor.noisy_q_arm2tgt
        obs = self.get_obs()
        info = self.compose_info()
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # Apply action
        self.low_level_control.update(action=np.array(action))
        # Step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        # Update and evaluate state
        terminated = self.done
        obs = self.get_obs()
        info = self.compose_info()
        reward = self.reward.compute(self.world.ur_arm.X_arm2plug, self.world.vrt_tgt.X_arm2tgt, terminated)
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.world.disconnect()

    def get_obs(self) -> npt.NDArray[np.float32]:
        # Build observation
        # Translation plug to target in arm frame
        noisy_p_plug2target = (self.noisy_p_arm2tgt - self.plug_sensor.noisy_p_arm2sensor).xyz
        # Minimal rotation plug to target
        noisy_q_plug2tgt = rp_math.quaternion_difference(self.plug_sensor.noisy_q_arm2sensor, self.target_sensor.noisy_q_arm2tgt).wxyz
        # Glue observation together
        obs = np.array((noisy_p_plug2target + noisy_q_plug2tgt), dtype=np.float32)
        return obs

    def compose_info(self) -> dict[str, Any]:
        # Calculate evaluation metrics
        p_plug2target = (self.world.vrt_tgt.p_arm2tgt - self.world.ur_arm.p_arm2plug).xyz
        q_arm2tgt = np.array(self.world.vrt_tgt.q_arm2tgt.wxyz)
        q_arm2plug = np.array(self.world.ur_arm.q_arm2plug.wxyz)
        self.task_pos_error = np.sqrt(np.sum(np.square(p_plug2target)))
        self.task_ang_error = np.arccos(np.clip((2 * (q_arm2tgt.dot(q_arm2plug))**2 - 1), -1.0, 1.0))
        info = {
            'error_pos': self.task_pos_error,
            'error_ang': self.task_ang_error,
            'solved': self.solved,
        }
        return info
