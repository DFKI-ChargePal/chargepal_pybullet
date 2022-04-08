""" This file defines the normalized distance-speed reward class. """
import logging
import copy

# mypy
import numpy as np
from typing import Dict, Any

from gym_chargepal.reward.config import NORMALIZED_DIST_SPEED_REWARD


LOGGER = logging.getLogger(__name__)


class NormalizedDistanceSpeedReward(object):
    """ Normalized Reward Class
    Penalizes high distances and high speeds.
    Reward calculation regarding:
    https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(NORMALIZED_DIST_SPEED_REWARD)
        config.update(hyperparams)
        self._hyperparams = config
        # params
        self._wd: float = self._hyperparams['w_dist']
        self._ws: float = self._hyperparams['w_speed']
        self._lower_db: float = self._hyperparams['lower_d_bound']
        self._lower_sb: float = self._hyperparams['lower_s_bound']
        self._dst_exp: float = self._hyperparams['dst_exp']
        self._final_dst_exp: float = self._hyperparams['final_dst_exp']

    def eval(self, dist: np.ndarray, speed: np.ndarray, done: bool, solved: bool) -> float:
        # l1-norm of the distance and speed
        dist_norm = min(np.mean(np.abs(dist)) * self._wd, 1.0)
        speed_norm = min(np.mean(np.abs(speed)) * self._ws, 1.0)
        if solved:
            # discrete task completion reward
            reward = 1.0
        else:
            if done:
                # final reward if task is not solved
                dist_reward = 1.0 - (dist_norm ** self._final_dst_exp)
            else:
                # calculate reward during episode
                dist_reward = 1.0 - (dist_norm ** self._dst_exp)
            speed_base = 1.0 - max(speed_norm, self._lower_sb)
            speed_exp = 1.0 / max(dist_norm, self._lower_sb)
            speed_discount = speed_base**speed_exp
            reward = speed_discount*dist_reward

        return float(reward)
