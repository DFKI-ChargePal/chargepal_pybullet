# global
import numpy as np
from rigmopy import utils_math as rp_math
from rigmopy import Pose, Quaternion, Vector3d

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
        config_start = ch.search(kwargs, 'start')
        config_target = ch.search(kwargs, 'target')
        config_reward = ch.search(kwargs, 'reward')
        config_jacobian = ch.search(kwargs, 'jacobian')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_joint_sensor = ch.search(kwargs, 'joint_sensor')
        config_target_sensor = ch.search(kwargs, 'target_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')
        # Placeholder noisy target sensor state
        self.x_arm2socket = Vector3d()
        self.q_arm2socket = Quaternion()
        config_low_level_control['plug_lin_config'] = self.x0_WP.xyz
        config_low_level_control['plug_ang_config'] = self.q0_WP.to_euler_angle()
        # Components
        self.world = WorldReacher(config_world, config_ur_arm, config_start, config_target)
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
        joint_config_0 = self.ik_solver.solve(self.world.sample_X0())
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
        X_tcp = Pose().from_pq(self.plug_sensor.noisy_p_arm2sensor, self.plug_sensor.noisy_q_arm2sensor)
        X_tgt = Pose().from_pq(self.plug_sensor.noisy_p_arm2sensor, self.plug_sensor.noisy_q_arm2sensor)
        V_tcp = self.plug_sensor.noisy_twist
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
        x_PW = self.plug_sensor.noisy_p_arm2sensor
        x_SW = self.target_sensor.noisy_p_arm2tgt
        x_SP = (x_SW - x_PW).xyz
        q_PW = self.plug_sensor.noisy_q_arm2sensor
        q_SW = self.target_sensor.noisy_q_arm2tgt
        q_SP = rp_math.quaternion_difference(q_PW, q_SW).wxyz
        # Get velocity signal
        i_twist_PW = self.plug_sensor.noisy_twist.xyzXYZ
        # Glue observation together
        obs = np.array((x_SP + q_SP + i_twist_PW), dtype=np.float32)
        # Calculate evaluation metrics
        q_SW_ = np.array(q_SW.wxyz)
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
