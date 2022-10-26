""" This file defines the controllers base class. """
# global
import abc
from dataclasses import dataclass

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Dict, Any


@dataclass
class ControllerCfg(ConfigHandler):
    wa_lin: float = 1.0  # action scaling in linear directions [m]
    wa_ang: float = 1.0  # action scaling in angular directions [rad]


class Controller(metaclass=abc.ABCMeta):
    """ Controller superclass. """
    def __init__(self, config: Dict[str, Any]):
        # Create configuration and override values
        self.cfg = ControllerCfg()
        self.cfg.update(**config)
