# global
import numpy as np
import pybullet as p
import quaternionic as quat

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.envs.env_base import Environment
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.worlds.world_plugger import WorldPlugger
from gym_chargepal.reward.reward_dist import DistanceReward
from gym_chargepal.sensors.sensor_socket import SocketSensor
from gym_chargepal.utility.tf import Quaternion, Translation, Pose
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
        
        # extract component hyperparameter from kwargs
        config_world = ch.search(kwargs, 'world')
        config_ur_arm = ch.search(kwargs, 'ur_arm')
        config_socket = ch.search(kwargs, 'socket')
        config_reward = ch.search(kwargs, 'reward')
        config_ik_solver = ch.search(kwargs, 'ik_solver')
        config_plug_sensor = ch.search(kwargs, 'plug_sensor')
        config_socket_sensor = ch.search(kwargs, 'socket_sensor')
        config_control_interface = ch.search(kwargs, 'control_interface')
        config_low_level_control = ch.search(kwargs, 'low_level_control')

        # start configuration in world coordinates
        self.x0_WP: Tuple[float, ...] = tuple(self.cfg.target_config.pos.as_array() + self.cfg.start_config.pos.as_array())
        q0_SP= self.cfg.start_config.ori.as_quaternionic()
        q0_WS = self.cfg.target_config.ori.as_quaternionic()
        q0_WP = tuple((q0_WS * q0_SP).ndarray)
        self.q0_WP = Quaternion(*q0_WP)

        # resolve cross references
        config_world['ur_arm'] = config_ur_arm
        config_world['socket'] = config_socket
        config_low_level_control['plug_lin_config'] = self.x0_WP
        config_low_level_control['plug_ang_config'] = p.getEulerFromQuaternion(self.q0_WP.as_tuple(order='xyzw'))

        # render option can be enabled with render() function
        self.is_render = False
        self.toggle_render_mode = False

        # components
        self.world = WorldPlugger(config_world)
        self.ik_solver = IKSolver(config_ik_solver, self.world.ur_arm)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world.ur_arm)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world.ur_arm)
        self.socket_sensor = SocketSensor(config_socket_sensor, self.world.socket)
        self.low_level_control = TcpPositionController(
            config_low_level_control,
            self.ik_solver,
            self.control_interface,
            self.plug_sensor
        )
        self.reward = DistanceReward(config_reward, self.clock)

    def reset(self) -> npt.NDArray[np.float32]:
        # reset environment
        self.clock.reset()

        if self.toggle_render_mode:
            self.world.disconnect()
            self.toggle_render_mode = False

        # reset robot by default joint configuration
        self.world.reset(render=self.is_render)

        # get start joint configuration by inverse kinematic
        mean_q0_WP = self.q0_WP.as_quaternionic()
        cov_q0_WP = quat.array(self.reset_rnd_gen.rand_quat(order='wxyz'))
        q0_WP = tuple((mean_q0_WP * cov_q0_WP).ndarray)
        ori = Quaternion(*q0_WP).as_tuple(order='xyzw')
        pos: Tuple[float, ...] = tuple(self.x0_WP + self.reset_rnd_gen.rand_linear())
        joint_config_0 = self.ik_solver.solve((pos, ori))

        # reset robot again
        self.world.reset(joint_config_0)

        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # apply action
        self.low_level_control.update(action=np.array(action))
        
        # step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        # Get new observation
        obs = self.get_obs()
        # Evaluate environment
        done = self.done
        X_tcp = Pose(
            Translation(*self.plug_sensor.get_pos()), 
            Quaternion(*(self.plug_sensor.get_ori()) + ('xyzw',))
            )
        X_tgt = Pose(
            Translation(*self.socket_sensor.get_pos()),
            Quaternion(*(self.socket_sensor.get_ori()) + ('xyzw',))
            )
        reward = self.reward.compute(X_tcp, X_tgt, done)
        info = self.compose_info()
        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        self.toggle_render_mode = True if not self.is_render else False
        self.is_render = True

    def close(self) -> None:
        self.world.disconnect()

    def get_obs(self) -> npt.NDArray[np.float32]:
        tgt_pos = np.array(self.socket_sensor.get_pos())
        plg_pos = np.array(self.plug_sensor.get_pos())
        dif_pos: Tuple[float, ...] = tuple(tgt_pos - plg_pos)

        tgt_ori = self.socket_sensor.get_ori()
        plg_ori = self.plug_sensor.get_ori()
        dif_ori = self.world.bullet_client.getDifferenceQuaternion(plg_ori, tgt_ori)
        obs = np.array((dif_pos + dif_ori), dtype=np.float32)

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
