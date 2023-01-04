""" This file defines the reward class for distances between two frames. """
# global
import numpy as np
from rigmopy import Pose
from dataclasses import dataclass

# local
from gym_chargepal.utility.env_clock import EnvironmentClock
from gym_chargepal.reward.reward import RewardCfg, Reward

# mypy
from numpy import typing as npt
from typing import Any, Dict


@dataclass
class DistanceRewardCfg(RewardCfg):
    spatial_pt_distance: float = 1.0
    dist_weight: float = 1.0
    exp_weight: float = 0.4


class DistanceReward(Reward):
    """ Evaluation class for spatial distances. """
    def __init__(self, config: Dict[str, Any], env_clock: EnvironmentClock):
        # Call super class
        super().__init__(config=config, env_clock=env_clock)
        # Create configuration and override values
        self.cfg: DistanceRewardCfg = DistanceRewardCfg()
        self.cfg.update(**config)

    def compute(self, X_tcp: Pose, X_tgt: Pose, done: bool) -> float:
        """ Compute state action reward """
        # Convert pose to sets of points
        pts_tcp = X_tcp.to_3pt_set(dist=self.cfg.spatial_pt_distance, axes='xy')
        pts_tgt = X_tgt.to_3pt_set(dist=self.cfg.spatial_pt_distance, axes='xy')
        # Distance between end-effector and target points
        distances: npt.NDArray[np.float32] = pts_tgt - pts_tcp
        # L1-norm of the distance
        dist_norm = min(np.mean(np.abs(distances)) * self.cfg.dist_weight, 1.0)
        # Calculate reward and scale it exponential
        r: float = 1.0 - (dist_norm ** self.cfg.exp_weight)
        reward = r if done else r / self.clock.t_end
        return reward
