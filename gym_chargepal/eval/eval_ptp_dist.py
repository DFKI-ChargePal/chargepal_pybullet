""" This file defines the evaluation class for the Point-to-Point environment with position controller. """
import copy
import numpy as np

from gym_chargepal.eval.config import EVAL_PTP_DIST
from gym_chargepal.eval.eval_ptp import EvalPtP

# mypy
from numpy import typing as npt
from typing import Any, Dict, Union

from gym_chargepal.utility.env_clock import EnvironmentClock
from gym_chargepal.sensors.sensor_virtual_ptp import VirtualTargetSensor
from gym_chargepal.sensors.sensor_virtual_adpstd import VirtualAdapterStationSensor
from gym_chargepal.sensors.sensor_virtual_plug import VirtualPlugSensor


class EvalDistancePtP(EvalPtP):
    """ Evaluation subclass class """
    def __init__(
        self, 
        hyperparams: Dict[str, Any], 
        clock: EnvironmentClock,
        target_sensor: Union[VirtualTargetSensor, VirtualAdapterStationSensor], 
        plug_sensor: VirtualPlugSensor
        ):
        config: Dict[str, Any] = copy.deepcopy(EVAL_PTP_DIST)
        config.update(hyperparams)
        EvalPtP.__init__(self, config, clock)

        self.target_sensor = target_sensor
        self.plug_sensor = plug_sensor

    def eval_reward(self) -> float:
        """ Calculate reward for small distance errors """
        # get virtual frame values
        tgt_ref_pos = np.array(self.target_sensor.get_pos_list(), dtype=np.float32)
        plg_ref_pos = np.array(self.plug_sensor.get_pos_list(), dtype=np.float32)
        # distance between tool and target
        distances: npt.NDArray[np.float32] = tgt_ref_pos - plg_ref_pos
        # l1-norm of the distance
        dist_norm = min(np.mean(np.abs(distances)) * self.hyperparams['distance_weight'], 1.0)
        # calculate reward and scale it exponential
        r: float = 1.0 - (dist_norm ** self.hyperparams['distance_exp'])
        reward = r if self.eval_done() else r / self.clock.t_end
        return reward
