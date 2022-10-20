""" This file defines the reward class for distances between two frames weighted by the TCP speed. """
# global
import numpy as np
from dataclasses import dataclass

# local
from gym_chargepal.utility.tf import Twist, Pose
from gym_chargepal.utility.cfg_handler import ConfigHandler
from gym_chargepal.utility.env_clock import EnvironmentClock

# mypy
from numpy import typing as npt
from typing import Any, Dict


@dataclass
class DistanceSpeedRewardCfg(ConfigHandler):
    spatial_pt_distance: float = 1.0
    dist_weight: float = 1.0
    dist_bound: float = 0.1
    speed_weight: float = 0.1
    speed_bound: float = 0.001
    exp_weight: float = 0.4


class DistanceSpeedReward:
    """ Reward class for speed weighted spatial distances. """
    def __init__(self, config: Dict[str, Any], env_clock: EnvironmentClock) -> None:
        # Create configuration and override values
        self.cfg = DistanceSpeedRewardCfg()
        self.cfg.update(**config)
        # Save references
        self.clock = env_clock

    def compute(self, X_tcp: Pose, V_tcp: Twist, X_tgt: Pose, done: bool) -> float:
        """ 
        Compute state action reward
            Reward calculation regarding: 
            https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
        """
        # Convert pose to sets of points
        pts_tcp = X_tcp.to_3pt_set(dist=self.cfg.spatial_pt_distance, axes='xy')
        pts_tgt = X_tgt.to_3pt_set(dist=self.cfg.spatial_pt_distance, axes='xy')
        # Distance between end-effector and target points
        distances: npt.NDArray[np.float32] = pts_tgt - pts_tcp
        speed = V_tcp.as_array()
        # Convert angular velocities into tangential speed with r = 1m
        speed[3:] = 2 * np.pi * speed[3:]
        # Compute L1-norm of speed and distance vector
        speed_norm = min(np.mean(np.abs(speed)) * self.cfg.speed_weight, 1.0)
        dist_norm = min(np.mean(np.abs(distances)) * self.cfg.dist_weight, 1.0)
        # Calculate discounted reward and scale it exponential
        r: float = 1.0 - (dist_norm ** self.cfg.exp_weight)
        if done:
            reward = r
        else:
            r = r / self.clock.t_end
            s_base = 1.0 - max(speed_norm, self.cfg.speed_bound)
            s_exp = 1.0 / max(dist_norm, self.cfg.dist_bound)
            speed_discount = s_base ** s_exp
            reward = speed_discount * r
        return reward