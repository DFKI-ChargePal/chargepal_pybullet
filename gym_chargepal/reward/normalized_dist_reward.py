""" This file defines the normalized distance reward class. """
import logging
import copy

# mypy
import numpy as np
from typing import Dict, Any

from gym_chargepal.reward.config import NORMALIZED_DIST_REWARD


LOGGER = logging.getLogger(__name__)


class NormalizedDistanceReward(object):
    """ Normalized Reward Class
    Penalizes high distances.
    Reward calculation regarding:
    https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(NORMALIZED_DIST_REWARD)
        config.update(hyperparams)
        self._hyperparams = config
        # params
        self._wd: float = self._hyperparams['w_dist']
        self._dst_exp: float = self._hyperparams['dst_exp']
        self._final_dst_exp: float = self._hyperparams['final_dst_exp']

    def eval(self, dist: np.ndarray, done: bool, solved: bool) -> float:
        # l1-norm of the distance
        dist_norm = min(np.mean(np.abs(dist)) * self._wd, 1.0)
        if solved:
            # discrete task completion reward
            reward = 1.0
        else:
            if done:
                # final reward if task is not solved
                reward = 1.0 - (dist_norm ** self._final_dst_exp)
            else:
                # calculate reward
                reward = 1.0 - (dist_norm ** self._dst_exp)
        return float(reward)
