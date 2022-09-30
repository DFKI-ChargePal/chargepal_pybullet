""" This file defines the evaluation class for the Point-to-Point environment with position controller. """
import copy
import numpy as np

from gym_chargepal.eval.config import EVAL_PTP_SPEED
from gym_chargepal.eval.eval_ptp import EvalPtP

# mypy
from numpy import typing as npt
from typing import Any, Dict

from gym_chargepal.utility.env_clock import EnvironmentClock
from gym_chargepal.sensors.sensor_virtual_ptp import VirtualTargetSensor
from gym_chargepal.sensors.sensor_virtual_plug import VirtualPlugSensor


class EvalSpeedPtP(EvalPtP):
    """ Evalation subclass class """
    def __init__(
        self, 
        hyperparams: Dict[str, Any], 
        clock: EnvironmentClock,
        target_sensor: VirtualTargetSensor, 
        plug_sensor: VirtualPlugSensor
        ):
        config: Dict[str, Any] = copy.deepcopy(EVAL_PTP_SPEED)
        config.update(hyperparams)
        EvalPtP.__init__(self, config, clock)
        # sensor references
        self.target_sensor = target_sensor
        self.plug_sensor = plug_sensor

    def eval_reward(self) -> float:
        """ Calculate reward for small distance errors 
        Reward calculation regarding: 
        https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0
        """
        # get virtual frame values
        tgt_ref_pos = np.array(self.target_sensor.get_pos_list(), dtype=np.float32)
        plg_ref_pos = np.array(self.plug_sensor.get_pos_list(), dtype=np.float32)
        speed = np.array(self.plug_sensor.get_vel_list(), dtype=np.float32)
        
        # distance between tool and target
        distances: npt.NDArray[np.float32] = tgt_ref_pos - plg_ref_pos
        # l1-norm of the speed and distance
        dist_norm = min(np.mean(np.abs(distances)) * self.hyperparams['distance_weight'], 1.0)
        speed_norm = min(np.mean(np.abs(speed)) * self.hyperparams['speed_weight'], 1.0)
        # calculate discounted reward and scale it exponential
        r: float = 1.0 - (dist_norm ** self.hyperparams['distance_exp'])
        if self.eval_done():
            reward = r
        else:
            r = r / self.clock.t_end
            s_base = 1.0 - max(speed_norm, self.hyperparams['speed_bound'])
            s_exp = 1.0 / max(dist_norm, self.hyperparams['distance_bound'])
            speed_discount = s_base ** s_exp
            reward = speed_discount * r

        return reward
