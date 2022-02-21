""" This file defines an environment with a point to point task """
import copy
import numpy as np

# mypy
from typing import Dict, Any, Tuple, List

from chargepal_pybullet.rl_gym.envs.env import Environment
from chargepal_pybullet.rl_gym.envs.config import ENVIRONMENT_PTP_1DOF_CARTESIAN_POSITION_CONTROL
# components
from chargepal_pybullet.rl_gym.worlds.world_ptp import WorldPoint2Point
from chargepal_pybullet.rl_gym.bullet.ik_solver import IKSolver
from chargepal_pybullet.rl_gym.bullet.joint_position_motor_control import JointPositionMotorControl
from chargepal_pybullet.rl_gym.controllers.controller_pos_1dof_cartesian import Position1dofCartesianController
from chargepal_pybullet.rl_gym.sensors.sensor_target_ptp import TargetSensor
from chargepal_pybullet.rl_gym.sensors.sensor_tool import ToolSensor
from chargepal_pybullet.rl_gym.reward.normalized_dist_reward import NormalizedDistanceReward


class EnvironmentP2PCartesian1DPositionCtrl(Environment):
    """ Cartesian Environment with 3 dof position controller - Task: point to point """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(ENVIRONMENT_PTP_1DOF_CARTESIAN_POSITION_CONTROL)
        config.update(hyperparams['common'])
        Environment.__init__(self, config)
        # components
        hp = hyperparams['components']
        hp['worlds']['gui'] = self._hyperparams['gui']
        self._world = WorldPoint2Point(hp['worlds'])
        self._ik_solver = IKSolver(hp['ik_solver'], self._world)
        self._control_interface = JointPositionMotorControl(hp['control_interface'], self._world)
        self._tool_sensor = ToolSensor(hp['tool_sensor'], self._world)
        self._target_sensor = TargetSensor(hp['target_sensor'], self._world)
        self._low_level_control = Position1dofCartesianController(
            hp['low_level_control'], self._ik_solver, self._control_interface, self._tool_sensor
        )
        self._reward = NormalizedDistanceReward(hp['reward'])
        # params
        self._tool_ori0 = self._hyperparams['start_ori']
        self._start_off = np.array(self._hyperparams['start_off'])
        self._start_var = np.array(self._hyperparams['start_var'])
        self._tgt_tolerance = self._hyperparams['tgt_tolerance']
        # logging
        self._info: Dict[str, Any] = {}
        # history
        # self.history: List[List[float]] = []

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
        done = True if self._n_step >= self._hyperparams['T'] else False
        reward = self._get_reward(done, solved)

        # log useful information
        self._info['done'] = done
        self._info['solved'] = solved
        self._info.update(info)

        # state, action, reward history
        # hist = [float(self._n_step), obs[1], float(action), float(reward)]
        # self.history.append(hist)

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        # draw target marker
        self._world.draw_target()
        # reset environment
        self._n_step = 0
        # reset robot
        self._world.reset()

        # solve ik to get joint configuration
        tgt = self._target_sensor.get_pos()
        p0 = tuple(tgt + self._start_off + self._start_var * np.random.randn(3))
        # p0 = tuple(tgt + self._start_off)
        pose0 = (p0, self._tool_ori0)
        joint_configuration = self._ik_solver.solve(pose0)

        # reset robot again
        self._world.reset(joint_configuration)

        # update sensors states
        self._update_sensors()
        return self._get_obs()[0]

    def exit(self) -> None:
        # np.savetxt('history.out', np.array(self.history, dtype=np.float32), delimiter=',', fmt='%.5f')
        self._world.exit()

    def get_info(self) -> Dict[str, Any]:
        return self._info

    def _update_sensors(self) -> None:
        self._tool_sensor.update()
        self._target_sensor.update()

    def _get_obs(self) -> Tuple[np.ndarray, Dict[str, float]]:
        tgt = np.array(self._target_sensor.get_pos())
        ee_pos = np.array(self._tool_sensor.get_pos())
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
