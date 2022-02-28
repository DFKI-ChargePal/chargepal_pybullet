""" This file defines the normalized reward class. """
import logging
import copy

# mypy
import numpy as np
from typing import Dict, Any

from gym_chargepal.reward.config import NORMALIZED_DIST_SPEED_REWARD


LOGGER = logging.getLogger(__name__)


class NormalizedDistSpeedReward(object):
    """ Normalized Reward Class
    Penalizes high distances and high velocities.
    Reward calculation regarding:
    https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(NORMALIZED_DIST_SPEED_REWARD)
        config.update(hyperparams)
        self._hyperparams = config
        # params
        self._T: int = self._hyperparams['T']
        self._wd: float = self._hyperparams['w_dist']
        self._ws: float = self._hyperparams['w_speed']
        self._dst_exp: float = self._hyperparams['dst_exp']
        self._lower_db: float = self._hyperparams['lower_d_bound']
        self._lower_sb: float = self._hyperparams['lower_s_bound']

    def eval(self, dist: np.ndarray, speed: np.ndarray, done: bool, solved: bool) -> float:
        # normalize and scale
        d = np.mean(np.abs(dist), axis=0)
        s = np.mean(np.abs(speed), axis=0)
        # l1-norm of the distance
        dist_norm = min(np.mean(d) * self._wd, 1.0)
        # l1-norm of the velocity
        speed_norm = min(np.mean(s) * self._ws, 1.0)

        if done:
            if solved:
                reward = 1.0 / self._T
            else:
                reward = (1.0 - dist_norm**self._dst_exp) / self._T
        else:
            # calculate reward
            d_rwd = 1.0 - dist_norm**self._dst_exp
            s_base = 1.0 - max(speed_norm, self._lower_sb)
            s_exp = 1.0 / max(dist_norm, self._lower_db)
            vel_discount = s_base**s_exp
            reward = (vel_discount * d_rwd) / self._T

        return reward
