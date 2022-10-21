# global
import numpy as np
import pybullet as p
import quaternionic as quat

# local
from gym_chargepal.envs.env import Environment
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.sensors.sensor_target_ptp import VirtTgtSensor
from gym_chargepal.controllers.controller_tcp_pos import TcpPositionController
from gym_chargepal.reward.reward_dist import DistanceReward
from gym_chargepal.utility.tf import Quaternion, Translation, Pose

# MyPy
from numpy import typing as npt
from typing import Callable, Dict, Any, Tuple


class EnvironmentTcpPositionCtrlPtP(Environment):
    """ Cartesian Environment with position controller - Task: point to point """
    def __init__(self, **kwargs: Dict[str, Any]):
        # Update environment configuration
        config_env = {} if 'config_env' not in kwargs else kwargs['config_env']
        Environment.__init__(self, config_env)

        # extract component hyperparameter from kwargs
        extract_config: Callable[[str], Dict[str, Any]] = lambda name: {} if name not in kwargs else kwargs[name]
        config_reward = extract_config('config_reward')
        config_world = extract_config('config_world')
        config_ik_solver = extract_config('config_ik_solver')
        config_control_interface = extract_config('config_control_interface')
        config_low_level_control = extract_config('config_low_level_control')
        config_plug_sensor = extract_config('config_plug_sensor')
        config_target_sensor = extract_config('config_target_sensor')

        # start configuration in world coordinates
        self.x0_WP: Tuple[float, ...] = tuple(self.cfg.target_config.pos.as_array() + self.cfg.start_config.pos.as_array())
        q0_SP= self.cfg.start_config.ori.as_quaternionic()
        q0_WS = self.cfg.target_config.ori.as_quaternionic()
        q0_WP = tuple((q0_WS * q0_SP).ndarray)
        self.q0_WP = Quaternion(*q0_WP)

        # resolve cross references
        config_world['target_pos'] = self.cfg.target_config.pos.as_tuple()
        config_world['target_ori'] = self.cfg.target_config.ori.as_tuple(order='xyzw')

        config_low_level_control['plug_lin_config'] = self.x0_WP
        config_low_level_control['plug_ang_config'] = p.getEulerFromQuaternion(self.q0_WP.as_tuple(order='xyzw'))

        # render option can be enabled with render() function
        self.is_render = False
        self.toggle_render_mode = False

        # components
        self.world = WorldPoint2Point(config_world)
        self.ik_solver = IKSolver(config_ik_solver, self.world)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world)
        self.target_sensor = VirtTgtSensor(config_target_sensor, self.world)
        self._low_level_control = TcpPositionController(
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

        # update sensors states
        self.update_sensors(target_sensor=True)
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # apply action
        self._low_level_control.update(action=np.array(action))
        
        # step simulation
        self.world.step(render=self.is_render)
        self.clock.tick()
        # update states
        self.update_sensors()
        obs = self.get_obs()

        # evaluate environment
        done = self.done
        X_tcp = Pose(
            Translation(*self.plug_sensor.get_pos()), 
            Quaternion(*(self.plug_sensor.get_ori()) + ('xyzw',))
            )
        X_tgt = Pose(
            Translation(*self.target_sensor.get_pos()),
            Quaternion(*(self.target_sensor.get_ori()) + ('xyzw',))
            )
        reward = self.reward.compute(X_tcp, X_tgt, done)
        info = self.compose_info()
        
        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        self.toggle_render_mode = True if not self.is_render else False
        self.is_render = True

    def close(self) -> None:
        self.world.disconnect()

    def update_sensors(self, target_sensor: bool=False) -> None:
        if target_sensor:
            self.target_sensor.update()
        self.plug_sensor.update()

    def get_obs(self) -> npt.NDArray[np.float32]:
        tgt_pos = np.array(self.target_sensor.get_pos())
        plg_pos = np.array(self.plug_sensor.get_pos())
        dif_pos: Tuple[float, ...] = tuple(tgt_pos - plg_pos)

        tgt_ori = self.target_sensor.get_ori()
        plg_ori = self.plug_sensor.get_ori()
        dif_ori = self.world.bullet_client.getDifferenceQuaternion(plg_ori, tgt_ori)
        obs = np.array((dif_pos + dif_ori), dtype=np.float32)

        tgt_ori_ = np.array(tgt_ori)
        plg_ori_ = np.array(plg_ori)
        self.error_pos = np.sqrt(np.sum(np.square(dif_pos)))
        self.error_ang = np.arccos(np.clip((2 * (tgt_ori_.dot(plg_ori_))**2 - 1), -1.0, 1.0))
        return obs

    def compose_info(self) -> Dict[str, Any]:
        info = {
            'error_pos': self.error_pos,
            'error_ang': self.error_ang,
            'solved': self.solved,
        }
        return info
