""" This file defines the reward class for finding the socket """
from __future__ import annotations

# global
import numpy as np
from rigmopy import Pose
from dataclasses import dataclass

# local
from gym_chargepal.reward.reward import (
    RewardCfg, Reward
)
from gym_chargepal.utility.env_clock import EnvironmentClock

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class FinderRewardCfg(RewardCfg):
    action_weight: float = 0.001


class FinderReward(Reward):
    """ Evaluation class for finding the socket 

    Args:
        Reward: Abstract base class 
    """
    def __init__(self, config: dict[str, Any], env_clock: EnvironmentClock):
        # Call super class
        super().__init__(config, env_clock)
        # Create configuration and overwrite values
        self.cfg: FinderRewardCfg = FinderRewardCfg()
        self.cfg.update(**config)

    def compute(self, action: npt.NDArray[np.float32], X_tcp: Pose, X_tgt: Pose, done: bool) -> float:
        """ Compute state action reward """
        action_reward: float = - self.cfg.action_weight * np.sum(np.square(action))
        
        # Check if there is spatial progress in plugging direction
        # Shift target by 2 cm since we are not interested in full plug in
        # Distance to goal in plugging direction
        p_tcp2tgt_wrt_base = (X_tgt.p - X_tcp.p)
        p_tcp2tgt_wrt_tool = X_tcp.q.apply(p_tcp2tgt_wrt_base, inverse=True)
        dz = p_tcp2tgt_wrt_tool.xyz[2]
        if dz < 0.02:
            distance_reward = 1.0
        else:
            distance_reward = 0.0

        reward = action_reward  + distance_reward
        return reward
