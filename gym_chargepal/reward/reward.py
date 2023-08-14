""" This file defines the reward base class. """
from __future__ import annotations

#  global
import abc
from dataclasses import dataclass

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler
from gym_chargepal.utility.env_clock import EnvironmentClock

# mypy
from typing import Any


@dataclass
class RewardCfg(ConfigHandler):
    pass


class Reward(metaclass=abc.ABCMeta):
    """ Reward superclass. """

    def __init__(self, config: dict[str, Any], env_clock: EnvironmentClock):
        # Create configuration and override values
        self.cfg = RewardCfg()
        self.cfg.update(**config)
        # Safe references
        self.clock = env_clock

    # @abc.abstractmethod
    # def compute(self, *args: Any, **kwargs: Any) -> float:
    #     """ Compute the state action reward """
    #     raise NotImplementedError('Must be implemented in subclass.')
