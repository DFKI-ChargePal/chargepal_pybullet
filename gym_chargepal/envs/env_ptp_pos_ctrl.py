import numpy as np
import pybullet as p

from gym_chargepal.envs.env import Environment
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.sensors.sensor_virtual_plug import VirtualPlugSensor
from gym_chargepal.sensors.sensor_target_ptp import TargetSensor
from gym_chargepal.sensors.sensor_virtual_ptp import VirtualTargetSensor
from gym_chargepal.controllers.controller_tcp_pos import TcpPositionController
from gym_chargepal.eval.eval_ptp_dist import EvalDistancePtP

# MyPy
from typing import Dict, Any, Tuple


class EnvironmentTcpPositionCtrlPtP(Environment):
    """ Cartesian Environment with position controller - Task: point to point """
    def __init__(self, **kwargs: Any):
        # Update environment configuration
        config_env = {} if 'config_env' not in kwargs else kwargs['config_env']
        Environment.__init__(self, config_env)

        # extract component hyperparameter from kwargs
        config_eval = {} if 'config_eval' not in kwargs else kwargs['config_eval']
        config_world = {} if 'config_world' not in kwargs else kwargs['config_world']
        config_ik_solver = {} if 'config_ik_solver' not in kwargs else kwargs['config_ik_solver']
        config_control_interface = {} if 'config_control_interface' not in kwargs else kwargs['config_control_interface']
        config_low_level_control = {} if 'config_low_level_control' not in kwargs else kwargs['config_low_level_control']
        config_plug_sensor = {} if 'config_plug_sensor' not in kwargs else kwargs['config_plug_sensor']
        config_plug_ref_sensor = {} if 'config_plug_ref_sensor' not in kwargs else kwargs['config_plug_ref_sensor']
        config_target_sensor = {} if 'config_target_sensor' not in kwargs else kwargs['config_target_sensor']
        config_target_ref_sensor = {} if 'config_target_ref_sensor' not in kwargs else kwargs['config_target_ref_sensor']

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

        config_low_level_control['plug_lin_config'] = self.pos_w_0
        config_low_level_control['plug_ang_config'] = self.ang_w_0

        # render option can be enabled with render() function
        self.is_render = False
        self.toggle_render_mode = False

        # componets
        self.world = WorldPoint2Point(config_world)
        self.ik_solver = IKSolver(config_ik_solver, self.world)
        self.control_interface = JointPositionMotorControl(config_control_interface, self.world)
        self.plug_sensor = PlugSensor(config_plug_sensor, self.world)
        self.plug_ref_sensor = VirtualPlugSensor(config_plug_ref_sensor, self.world)
        self.target_sensor = TargetSensor(config_target_sensor, self.world)
        self.target_ref_sensor = VirtualTargetSensor(config_target_ref_sensor, self.world)
        self._low_level_control = TcpPositionController(
            config_low_level_control,
            self.ik_solver,
            self.control_interface,
            self.plug_sensor
        )
        self.eval = EvalDistancePtP(
            config_eval, 
            self.clock, 
            self.target_ref_sensor, 
            self.plug_ref_sensor
        )

        # logging
        self.error_pos = np.inf
        self.error_ang = np.inf

    def reset(self) -> np.ndarray:
        # reset environment
        self.clock.reset()

        if self.toggle_render_mode:
            self.world.disconnect()
            self.toggle_render_mode = False

        # reset robot by default joint configuration
        self.world.reset(render=self.is_render)

        # get start joint configuration by inverse kinematic
        pos_0 = tuple(np.array(self.pos_w_0) + np.array(self.pos_var_0) * self.rs.randn(3))  # type: ignore
        ang_0 = tuple(np.array(self.ang_w_0) + np.array(self.ang_var_0) * self.rs.randn(3))  # type: ignore
        ori_0 = p.getQuaternionFromEuler(ang_0, physicsClientId=self.world.physics_client_id)
        joint_config_0 = self.ik_solver.solve((pos_0, ori_0))

        # reset robot again
        self.world.reset(joint_config_0)

        # update sensors states
        self.update_sensors(target_sensor=True)
        return self.get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
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
        done = self.eval.eval_done()
        reward = self.eval.eval_reward()
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
            self.target_ref_sensor.update()
        self.plug_sensor.update()
        self.plug_ref_sensor.update()

    def get_obs(self) -> np.ndarray:
        tgt_pos = np.array(self.target_sensor.get_pos())
        plg_pos = np.array(self.plug_sensor.get_pos())
        dif_pos = tuple(tgt_pos - plg_pos)

        tgt_ori = self.target_sensor.get_ori()
        plg_ori = self.plug_sensor.get_ori()
        dif_ori = p.getDifferenceQuaternion(plg_ori, tgt_ori, physicsClientId=self.world.physics_client_id)
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
            'solved': self.eval.eval_solve(self.error_pos, self.error_ang),
        }
        return info
