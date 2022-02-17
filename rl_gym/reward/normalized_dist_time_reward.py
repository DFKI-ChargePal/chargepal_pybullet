""" This file defines the normalized reward class. """
import logging
import copy

# mypy
import numpy as np
from typing import Dict, Any

from gym_env.reward.config import NORMALIZED_DIST_TIME_REWARD


LOGGER = logging.getLogger(__name__)


class NormalizedDistanceTimeReward(object):
    """ Normalized Reward Class
    Penalizes high distances and long episodes.
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(NORMALIZED_DIST_TIME_REWARD)
        config.update(hyperparams)
        self._hyperparams = config
        # params
        self._T: int = self._hyperparams['T']
        self._wd: float = self._hyperparams['w_dist']
        self._dst_exp: float = self._hyperparams['dst_exp']

    def eval(self, dist: np.ndarray, ts: int, done: bool, solved: bool) -> float:
        # l1-norm of the distance
        dist_norm = min(np.sum(np.abs(dist)) * self._wd, 1.0)
        if done and not solved:
            # discrete reward
            reward = - 1.0 / self._T
        else:
            # calculate reward
            time_scale_exp = max(1.0 - ts/self._T, self._dst_exp)
            reward = (1.0 - dist_norm**time_scale_exp) / self._T
        return reward
