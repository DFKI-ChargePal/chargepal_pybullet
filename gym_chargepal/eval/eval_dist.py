""" This file defines the evaluation class for distances between two frames. """
# global
import copy
import numpy as np
import quaternionic as quat

# local
from gym_chargepal.eval.config import EVAL_DIST


# mypy
from numpy import typing as npt
from typing import Any, Dict, List, Tuple
from gym_chargepal.utility.env_clock import EnvironmentClock


class EvalDistance:
    """ Evaluation class for spatial distances. """
    def __init__(self, hyperparams: Dict[str, Any], env_clock: EnvironmentClock):
        config: Dict[str, Any] = copy.deepcopy(EVAL_DIST)
        config.update(hyperparams)
        self.clock = env_clock
        self.hyperparams = config

    def is_done(self) -> bool:
        return self.clock.check_for_time_horizon()

    def is_solved(self, scalar_pos_error: float, scalar_ang_err: float) -> bool:
        """ Check if error is small enough """
        eps_pos = self.hyperparams['task_epsilon_pos']
        eps_ang = self.hyperparams['task_epsilon_ang']
        return True if scalar_pos_error < eps_pos and scalar_ang_err < eps_ang else False

    def _pose_to_pts(
        self,
        pos: Tuple[float, ...], 
        ori: Tuple[float, ...], 
        dist: float = 0.1
        ) -> npt.NDArray[np.float32]:
        """ Transform a pose to 3 spatial points. """
        points: List[List[float]] = []
        # First spatial point is just the origin of the pose
        points.append(list(pos))
        # Bring pose in right form
        p1 = np.array(pos)
        q1 = quat.array((ori[3],) + ori[0:3])
        # Get second point as extension of the x-axis
        p2 = q1.rotate((dist, 0.0, 0.0)) + p1
        points.append(list(p2))
        # Get third point as extension of the y-axis
        p3 = q1.rotate((0.0, dist, 0.0)) + p1
        points.append(list(p3))        
        return np.array(points, dtype=np.float32)

    def calc_reward(
        self, 
        pos_ee: Tuple[float, ...], 
        ori_ee: Tuple[float, ...],
        pos_tg: Tuple[float, ...], 
        ori_tg: Tuple[float, ...]
        ) -> float:
        """ Calculate state action reward """
        points_ee = self._pose_to_pts(pos_ee, ori_ee, dist=1.0)
        points_tg = self._pose_to_pts(pos_tg, ori_tg, dist=1.0)
        # distance between end-effector and target points
        distances: npt.NDArray[np.float32] = points_tg - points_ee
        # l1-norm of the distance
        dist_norm = min(np.mean(np.abs(distances)) * self.hyperparams['distance_weight'], 1.0)
        # calculate reward and scale it exponential
        r: float = 1.0 - (dist_norm ** self.hyperparams['distance_exp'])
        reward = r if self.is_done() else r / self.clock.t_end
        return reward
