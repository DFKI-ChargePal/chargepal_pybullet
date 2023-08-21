""" This file defines the controllers base class. """
from __future__ import annotations

# global
import abc
import numpy as np
from dataclasses import dataclass

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Any
from numpy import typing as npt


@dataclass
class ControllerCfg(ConfigHandler):
    wa_lin: float = 1.0  # action scaling in linear directions [m]
    wa_ang: float = 1.0  # action scaling in angular directions [rad]


class Controller(metaclass=abc.ABCMeta):
    """ Controller superclass. """
    def __init__(self, config: dict[str, Any]):
        # Create configuration and override values
        self.cfg = ControllerCfg()
        self.cfg.update(**config)

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def update(self, action: npt.NDArray[np.float32]) -> None:
        raise NotImplementedError('Must be implemented in subclass.')
