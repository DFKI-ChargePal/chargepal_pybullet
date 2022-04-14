""" This file defines the evaluation class for Point-to-Point environments. """
import abc 
import copy

import numpy as np

from gym_chargepal.eval.config import EVAL_PTP
from gym_chargepal.utility.env_clock import EnvironmentClock

# mypy
from typing import (
    Any, 
    Dict,
)


class EvalPtP:
    """ Evalation parent class """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any], env_clock: EnvironmentClock):
        config: Dict[str, Any] = copy.deepcopy(EVAL_PTP)
        config.update(hyperparams)
        self.clock = env_clock
        self.hyperparams = config

    def eval_done(self) -> bool:
        return self.clock.is_at_time_horizon()

    def eval_solve(self, scalar_pos_error: float, scalar_ang_err: float) -> bool:
        """ Check if error is small enough """
        eps_pos = self.hyperparams['task_epsilon_pos']
        eps_ang = self.hyperparams['task_epsilon_ang']
        return True if scalar_pos_error < eps_pos and scalar_ang_err < eps_ang else False

    @abc.abstractmethod
    def eval_reward(self) -> float:
        """ Calculate state action reward """
        raise NotImplementedError('Must be implemented in subclass.')
