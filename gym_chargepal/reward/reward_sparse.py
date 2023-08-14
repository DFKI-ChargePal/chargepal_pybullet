""" This file defines the reward class for finding the socket """
from __future__ import annotations

# global
import numpy as np
from dataclasses import dataclass
from rigmopy import Pose, Vector3d, Vector6d

# local
from gym_chargepal.reward.reward import (
    RewardCfg, Reward
)
from gym_chargepal.utility.env_clock import EnvironmentClock

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class SparseFinderRewardCfg(RewardCfg):
    spatial_pt_distance: float = 1.0  # distance to create reference points
    pose_weight: float = 1.0


class SparseFinderReward(Reward):
    """ Evaluation class for finding the socket """
    def __init__(self, config: dict[str, Any], env_clock: EnvironmentClock):
        # Call super class
        super().__init__(config, env_clock)
        # Create configuration and overwrite values
        self.cfg: SparseFinderRewardCfg = SparseFinderRewardCfg()
        self.cfg.update(**config)

    def compute(self, solved: bool,  X_tcp: Pose, X_tgt: Pose, F_tcp: Vector6d, done: bool) -> float:
        """ Compute state action reward """
        # Force
        ft_vec = list(F_tcp.xyzXYZ)
        # Get force in plugging direction
        ft_z = ft_vec[2]
        # Check if force in plugging direction within 15% of maximal force
        if 0.0 < ft_z < 0.15:
            contact_reward = 1.0
        else:
            contact_reward = 0.0
        # For the the other directions penalize high forces. 
        # However, search for the force-torque outlier and negatively reward only that one.
        del ft_vec[2]
        ft_vec = [abs(ft) for ft in ft_vec]
        ft_max = max(ft_vec)
        force_reward = -ft_max
        # Reward if plug stay close to the target to avoid diverging
        p_tcp2tgt_wrt_base = (X_tgt.p - X_tcp.p)
        dist_norm = np.sum(np.square(p_tcp2tgt_wrt_base.xyz))
        if dist_norm > 0.05:
            div_reward = -1.0 * self.clock.t
        else:
            div_reward = 0.0
        # Check if there is spatial progress in plugging direction
        # Shift target by 2 cm since we are not interested in full plug in
        p_tcp2tgt_wrt_tool = X_tcp.q.apply(p_tcp2tgt_wrt_base, inverse=True)
        # Distance to goal in plugging direction
        dz = p_tcp2tgt_wrt_tool.xyz[2] - 0.02
        if dz < 0.02:
            distance_reward = 1.0 - (abs(dz) ** 0.4)
            contact_reward = 1.0
        else:
            distance_reward = 0.0
        # Gather reward signals
        if solved:
            reward = (1 / self.clock.t) * self.clock.t_end
        elif done:
            reward = distance_reward + div_reward
        else:
            reward = -1.0 + contact_reward + force_reward + distance_reward + div_reward

        return reward
