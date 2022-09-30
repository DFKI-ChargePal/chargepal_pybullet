import numpy as np
import pybullet as p

from gym_chargepal.envs.env import Environment
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.sensors.sensor_joints import JointSensor
from gym_chargepal.sensors.sensor_target_ptp import TargetSensor
from gym_chargepal.sensors.sensor_virtual_plug import VirtualPlugSensor
from gym_chargepal.sensors.sensor_virtual_ptp import VirtualTargetSensor
from gym_chargepal.controllers.controller_tcp_vel import TcpVelocityController
from gym_chargepal.eval.eval_ptp_speed import EvalSpeedPtP

# mypy
from numpy import typing as npt
from typing import Any, Callable, Dict, Tuple



class EnvironmentTcpVelocityCtrlPtP(Environment):
    """ Cartestian Environment with velocity controller - Task: point to point """
    def __init__(self, **kwargs: Dict[str, Any]):
        # Update environment configuration
        config_env = {} if 'config_env' not in kwargs else kwargs['config_env']
        Environment.__init__(self, config_env)
        
        # extract component hyperparameter from kwargs
        extract_config: Callable[[str], Dict[str, Any]] = lambda name: {} if name not in kwargs else kwargs[name]
        config_eval = extract_config('config_eval')
        config_world = extract_config('config_world')
        config_jacobian = extract_config('config_jacobian')
        config_ik_solver = extract_config('config_ik_solver')
        config_control_interface = extract_config('config_control_interface')
        config_low_level_control = extract_config('config_low_level_control')
        config_joint_sensor = extract_config('config_joint_sensor')
        config_plug_sensor = extract_config('config_plug_sensor')
        config_plug_ref_sensor = extract_config('config_plug_ref_sensor')
        config_target_sensor = extract_config('config_target_sensor')
        config_target_ref_sensor = extract_config('config_target_ref_sensor')

        # start configuration in world coordinates
        self.pos_w_0 = tuple(
            [sum(ep) for ep in zip(self.hyperparams['tgt_config_pos'], self.hyperparams['start_config_pos'])]
            )
        self.ang_w_0 = tuple(
            [sum(ep) for ep in zip(self.hyperparams['tgt_config_ang'], self.hyperparams['start_config_ang'])]
            )
        # reset variance
        self.pos_var_0 = self.hyperparams['reset_variance'][0]
        self.ang_var_0 = self.hyperparams['reset_variance'][1]

        # resolve cross references
        config_world['target_pos'] = self.hyperparams['tgt_config_pos']
        config_world['target_ori'] = p.getQuaternionFromEuler(self.hyperparams['tgt_config_ang'])

        # render option can be enabled with render() function
        self.is_render = False
        self.toggle_render_mode = False

        # components
        self._world = WorldPoint2Point(config_world)
        self._jacobian = Jacobian(config_jacobian, self._world)
        self._ik_solver = IKSolver(config_ik_solver, self._world)
        self._control_interface = JointVelocityMotorControl(config_control_interface, self._world)
        self._joint_sensor = JointSensor(config_joint_sensor, self._world)
        self._plug_sensor = PlugSensor(config_plug_sensor, self._world)
        self._plug_ref_sensor = VirtualPlugSensor(config_plug_ref_sensor, self._world)
        self._target_sensor = TargetSensor(config_target_sensor, self._world)
        self._target_ref_sensor = VirtualTargetSensor(config_target_ref_sensor, self._world)
        self._low_level_control = TcpVelocityController(
            config_low_level_control,
            self._jacobian, 
            self._control_interface,
            self._plug_sensor, 
            self._joint_sensor
        )
        self.eval = EvalSpeedPtP(
            config_eval,
            self.clock,
            self._target_ref_sensor,
            self._plug_ref_sensor
        )

        # logging
        self.error_pos = np.inf
        self.error_ang = np.inf

    def reset(self) -> npt.NDArray[np.float32]:
        # reset environment
        self.clock.reset()

        if self.toggle_render_mode:
            self._world.disconnect()
            self.toggle_render_mode = False

        # reset robot by default joint configuration
        self._world.reset(render=self.is_render)

        # get start joint configuration by inverse kinematic
        pos_0 = tuple(np.array(self.pos_w_0) + np.array(self.pos_var_0) * self.rs.randn(3))  # type: ignore
        ang_0 = tuple(np.array(self.ang_w_0) + np.array(self.ang_var_0) * self.rs.randn(3))  # type: ignore
        ori_0 = p.getQuaternionFromEuler(ang_0, physicsClientId=self._world.physics_client_id)
        joint_config_0 = self._ik_solver.solve((pos_0, ori_0))

        # reset robot again
        self._world.reset(joint_config_0)

        # update sensors states
        self.update_sensors(target_sensor=True)
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # apply action
        self._low_level_control.update(action=np.array(action))

        # step simulation
        self._world.step(render=self.is_render)
        self.clock.tick()
        # update states
        self.update_sensors()
        obs = self.get_obs()

        # evaluate environment
        done = self.eval.eval_done()
        reward = self.eval.eval_reward()
        info = self.compose_info()

        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        self.toggle_render_mode = True if not self.is_render else False
        self.is_render = True

    def close(self) -> None:
        self._world.disconnect()

    def update_sensors(self, target_sensor: bool=False) -> None:
        if target_sensor:
            self._target_sensor.update()
            self._target_ref_sensor.update()
        self._plug_sensor.update()
        self._plug_ref_sensor.update()
        self._joint_sensor.update()

    def get_obs(self) -> npt.NDArray[np.float32]:
        # get position signals
        tgt_pos = np.array(self._target_sensor.get_pos())
        plg_pos = np.array(self._plug_sensor.get_pos())
        dif_pos: Tuple[float, ...] = tuple(tgt_pos - plg_pos)

        tgt_ori = self._target_sensor.get_ori()
        plg_ori = self._plug_sensor.get_ori()
        dif_ori = p.getDifferenceQuaternion(plg_ori, tgt_ori, physicsClientId=self._world.physics_client_id)

        # get velocity signal
        lin_vel = self._plug_sensor.get_lin_vel()
        ang_vel = self._plug_sensor.get_ang_vel()

        # build observation
        obs = (dif_pos + dif_ori + lin_vel + ang_vel)
        obs_nd = np.array(obs, dtype=np.float32)

        tgt_ori_ = np.array(tgt_ori)
        plg_ori_ = np.array(plg_ori)
        self.error_pos = np.sqrt(np.sum(np.square(dif_pos)))
        self.error_ang = np.arccos(np.clip((2 * (tgt_ori_.dot(plg_ori_))**2 - 1), -1.0, 1.0))
        return obs_nd

    def compose_info(self) -> Dict[str, Any]:
        info = {
            'error_pos': self.error_pos,
            'error_ang': self.error_ang,
            'solved': self.eval.eval_solve(self.error_pos, self.error_ang),
        }
        return info
