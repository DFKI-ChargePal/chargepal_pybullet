""" This file defines an environment with a point to point task """
import copy
import numpy as np

# mypy
from typing import Dict, Any, Tuple, Optional

from gym_chargepal.envs.env import Environment
from gym_chargepal.envs.config import ENVIRONMENT_PTP_3DOF_CARTESIAN_POSITION_CONTROL
# components
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.controllers.controller_pos_3dof_cartesian import Position3dofCartesianController
from gym_chargepal.sensors.sensor_target_ptp import TargetSensor
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.reward.normalized_dist_reward import NormalizedDistanceReward


class EnvironmentP2PCartesian3DPositionCtrl(Environment):
    """ Cartesian Environment with 3 dof position controller - Task: point to point """
    def __init__(self, **kwargs: Any):
        # Update environment configuration
        config_env = {} if 'config_env' not in kwargs else kwargs['config_env']
        config: Dict[str, Any] = copy.deepcopy(
            ENVIRONMENT_PTP_3DOF_CARTESIAN_POSITION_CONTROL)
        config.update(config_env)
        Environment.__init__(self, config)

        # extract component hyperparameter from kwargs
        config_reward = {} if 'config_reward' not in kwargs else kwargs['config_reward']
        config_world = {} if 'config_world' not in kwargs else kwargs['config_world']
        config_ik_solver = {} if 'config_ik_solver' not in kwargs else kwargs['config_ik_solver']
        config_control_interface = {} if 'config_control_interface' not in kwargs else kwargs['config_control_interface']
        config_low_level_control = {} if 'config_low_level_control' not in kwargs else kwargs['config_low_level_control']
        config_plug_sensor = {} if 'config_plug_sensor' not in kwargs else kwargs['config_plug_sensor']
        config_target_sensor = {} if 'config_target_sensor' not in kwargs else kwargs['config_target_sensor']

        # resolve cross references
        config_world['gui'] = self._hyperparams['gui']
        config_low_level_control['home_orientation'] = self._hyperparams['start_ori']

        # componets
        self._world = WorldPoint2Point(config_world)
        self._ik_solver = IKSolver(config_ik_solver, self._world)
        self._control_interface = JointPositionMotorControl(config_control_interface, self._world)
        self._plug_sensor = PlugSensor(config_plug_sensor, self._world)
        self._target_sensor = TargetSensor(config_target_sensor, self._world)
        self._low_level_control = Position3dofCartesianController(
            config_low_level_control, 
            self._ik_solver, 
            self._control_interface, 
            self._plug_sensor
        )
        self._reward = NormalizedDistanceReward(config_reward)
        # params
        self._tool_ori0 = self._hyperparams['start_ori']
        self._start_off = np.array(self._hyperparams['start_off'])
        self._start_var = np.array(self._hyperparams['start_var'])
        self._tgt_tolerance = self._hyperparams['tgt_tolerance']
        # logging
        self._info: Dict[str, Any] = {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """ Execute environment/simulation step. """
        # apply action
        self._low_level_control.update(action=action)
        # step simulation
        self._world.step()
        self._n_step += 1
        # update states
        self._update_sensors()
        obs, info = self._get_obs()

        # evaluate environment
        solved = self._check_if_solved(d_lin=np.sqrt(np.sum(np.square(obs))))
        done = True if self._n_step >= self._hyperparams['T'] or solved else False
        reward = self._get_reward(done, solved)

        # log useful information
        self._info['done'] = done
        self._info['solved'] = solved
        self._info.update(info)

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        # reset environment
        self._n_step = 0
        # reset robot
        self._world.reset()

        # solve ik to get joint configuration
        tgt = self._target_sensor.get_pos()
        p0 = tuple(tgt + self._start_off + self._start_var * np.random.randn(3))
        pose0 = (p0, self._tool_ori0)
        joint_configuration = self._ik_solver.solve(pose0)

        # reset robot again
        self._world.reset(joint_configuration)

        # update sensors states
        self._update_sensors()
        return self._get_obs()[0]

    def close(self) -> None:
        self._world.disconnect()

    def get_info(self) -> Dict[str, Any]:
        return self._info

    def _update_sensors(self) -> None:
        self._plug_sensor.update()
        self._target_sensor.update()

    def _get_obs(self) -> Tuple[np.ndarray, Dict[str, float]]:
        tgt = np.array(self._target_sensor.get_pos())
        ee_pos = np.array(self._plug_sensor.get_pos())
        d_pos = np.array((tgt - ee_pos), dtype=np.float32)

        info = {
            'dist': np.sqrt(np.sum(np.square(d_pos)))
        }
        return d_pos, info

    def _get_reward(self, done: bool, solved: bool) -> float:
        # distance between tool and target
        p_diff, _ = self._get_obs()
        return self._reward.eval(p_diff, done=done, solved=solved)

    def _check_if_solved(self, d_lin: float) -> bool:
        solved = True if d_lin < self._tgt_tolerance else False
        return solved
