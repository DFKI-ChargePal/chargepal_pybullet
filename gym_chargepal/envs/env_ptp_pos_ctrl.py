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
from gym_chargepal.controllers.controller_pos_cartesian import PositionCartesianController
from gym_chargepal.reward.normalized_dist_reward import NormalizedDistanceReward

# MyPy
from typing import Dict, Any, Tuple


class EnvironmentPtPCartesianPositionCtrl(Environment):
    """ Cartesian Environment with 3 dof position controller - Task: point to point """
    def __init__(self, **kwargs: Any):
        # Update environment configuration
        config_env = {} if 'config_env' not in kwargs else kwargs['config_env']
        Environment.__init__(self, config_env)

        # extract component hyperparameter from kwargs
        config_reward = {} if 'config_reward' not in kwargs else kwargs['config_reward']
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
            [sum(ep) for ep in zip(self._hyperparams['tgt_config_pos'], self._hyperparams['start_config_pos'])]
            )
        self.ang_w_0 = tuple(
            [sum(ep) for ep in zip(self._hyperparams['tgt_config_ang'], self._hyperparams['start_config_ang'])]
            )
        # reset variance
        self.pos_var_0 = self._hyperparams['reset_variance'][0]
        self.ang_var_0 = self._hyperparams['reset_variance'][1]

        # resolve cross references
        config_world['target_pos'] = self._hyperparams['tgt_config_pos']
        config_world['target_ori'] = p.getQuaternionFromEuler(self._hyperparams['tgt_config_ang'])

        config_low_level_control['plug_lin_config'] = self.pos_w_0
        config_low_level_control['plug_ang_config'] = self.ang_w_0

        # render option can be enabled with render() function
        self.is_render = False
        self.toggle_render_mode = False

        # componets
        self._world = WorldPoint2Point(config_world)
        self._ik_solver = IKSolver(config_ik_solver, self._world)
        self._control_interface = JointPositionMotorControl(config_control_interface, self._world)
        self._plug_sensor = PlugSensor(config_plug_sensor, self._world)
        self._plug_ref_sensor = VirtualPlugSensor(config_plug_ref_sensor, self._world)
        self._target_sensor = TargetSensor(config_target_sensor, self._world)
        self._target_ref_sensor = VirtualTargetSensor(config_target_ref_sensor, self._world)
        self._low_level_control = PositionCartesianController(
            config_low_level_control,
            self._ik_solver,
            self._control_interface,
            self._plug_sensor
        )
        self._reward = NormalizedDistanceReward(config_reward)

        # logging
        self._error_pos = np.inf
        self._error_ang = np.inf
        self._done = False
        self._solved = False

    def reset(self) -> np.ndarray:
        # reset environment
        self._n_step = 0
        self._done = False
        self._solved = False

        if self.toggle_render_mode:
            self._world.disconnect()
            self.toggle_render_mode = False

        # reset robot by default joint configuration
        self._world.reset(render=self.is_render)

        # get start joint configuration by inverse kinematic
        pos_0 = tuple(np.array(self.pos_w_0) + np.array(self.pos_var_0) * np.random.randn(3))
        ang_0 = tuple(np.array(self.ang_w_0) + np.array(self.ang_var_0) * np.random.randn(3))
        ori_0 = p.getQuaternionFromEuler(ang_0, physicsClientId=self._world.physics_client_id)
        joint_config_0 = self._ik_solver.solve((pos_0, ori_0))

        # reset robot again
        self._world.reset(joint_config_0)

        # update sensors states
        self.update_sensors(target_sensor=True)
        return self.obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # apply action
        self._low_level_control.update(action=np.array(action))
        
        # step simulation
        self._world.step(render=self.is_render)
        self._n_step += 1
        # update states
        self.update_sensors()
        obs = self.obs()

        # evaluate environment
        reward = self.calc_reward()
        done = self._done
        info = self.get_info()
        
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

    def obs(self) -> np.ndarray:
        tgt_pos = np.array(self._target_sensor.get_pos())
        plg_pos = np.array(self._plug_sensor.get_pos())
        dif_pos = tuple(tgt_pos - plg_pos)

        tgt_ori = self._target_sensor.get_ori()
        plg_ori = self._plug_sensor.get_ori()
        dif_ori = p.getDifferenceQuaternion(plg_ori, tgt_ori, physicsClientId=self._world.physics_client_id)
        obs = np.array((dif_pos + dif_ori), dtype=np.float32)

        tgt_ori_ = np.array(tgt_ori)
        plg_ori_ = np.array(plg_ori)
        self._error_pos = np.sqrt(np.sum(np.square(dif_pos)))
        self._error_ang = np.arccos(2 * (tgt_ori_.dot(plg_ori_))**2 - 1)
        return obs

    def calc_reward(self) -> float:
        self.check_performance()
        # get virtual frame values
        tgt_ref_pos = np.array(self._target_ref_sensor.get_pos_list())
        plg_ref_pos = np.array(self._plug_ref_sensor.get_pos_list())
        # distance between tool and target
        diff_pos = tgt_ref_pos - plg_ref_pos
        return self._reward.eval(diff_pos, done=self._done, solved=self._solved)

    def check_performance(self) -> None:
        eps_pos = self._hyperparams['task_epsilon_pos']
        eps_ang = self._hyperparams['task_epsilon_ang']
        self._solved = True if self._error_pos < eps_pos and self._error_ang < eps_ang else False
        self._done = True if self._n_step >= self._hyperparams['T'] else False

    def get_info(self) -> Dict[str, Any]:
        info = {
            'error_pos': self._error_pos,
            'error_ang': self._error_ang,
            'done': self._done,
            'solved': self._solved,
        }
        return info
