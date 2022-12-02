# global
import numpy as np
from dataclasses import dataclass

# local
from gym_chargepal.utility.tf import Pose, Wrench
from gym_chargepal.reward.reward import RewardCfg, Reward
from gym_chargepal.utility.env_clock import EnvironmentClock

# typing
from typing import Any, Dict
from numpy import typing as npt


@dataclass
class PoseWrenchRewardCfg(RewardCfg):
    spatial_pt_distance: float = 1.0  # distance to create reference points
    pose_weight: float = 1.0
    pose_bound: float = 0.1
    wrench_weight: float = 0.01
    wrench_bound: float = 0.001
    exp_weight: float = 0.4


class PoseWrenchReward(Reward):
    """ Reward class that rewards small spatial errors and a small wrenching. """
    def __init__(self, config: Dict[str, Any], env_clock: EnvironmentClock):
        # Call super class
        super().__init__(config, env_clock)
        # Create configuration object and update values
        self.cfg: PoseWrenchRewardCfg = PoseWrenchRewardCfg()
        self.cfg.update(**config)

    def compute(self, X_tcp: Pose, X_tgt: Pose, F_tcp: Wrench, done: bool) -> float:
        """ Compute state action  reward """
        # Convert pose to sets of points
        pts_tcp = X_tcp.to_3pt_set(dist=self.cfg.spatial_pt_distance, axes='xy')
        pts_tgt = X_tgt.to_3pt_set(dist=self.cfg.spatial_pt_distance, axes='xy')
        # Distance between end-effector and target points
        distances: npt.NDArray[np.float32] = pts_tgt - pts_tcp
        # Search for the force-torque outlier and negatively reward only that one.
        ft_max = max(F_tcp.as_tuple())
        ft_max_weight = min(ft_max * self.cfg.wrench_weight, 1.0)
        # L1-norm of the distance
        dist_norm = min(np.mean(np.abs(distances)) * self.cfg.pose_weight, 1.0)
        # Calculate discounted reward and scale it exponential
        r: float = 1.0 - (dist_norm ** self.cfg.exp_weight)
        if ft_max >= 1.0:
            reward = 0.0
        elif done:
            reward = r
        else:
            r = r / self.clock.t_end
            ft_base = 1.0 - max(ft_max_weight, self.cfg.wrench_bound)
            ft_exp = 1.0 / max(dist_norm, self.cfg.pose_bound)
            ft_discount = ft_base ** ft_exp
            ft_discount = 1.0
            reward = ft_discount * r
        return reward
