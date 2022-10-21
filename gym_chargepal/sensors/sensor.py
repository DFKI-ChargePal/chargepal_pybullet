""" This file defines the sensors base class. """
# global
import abc
from dataclasses import dataclass

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Dict, Any


@dataclass
class SensorCfg(ConfigHandler):
    pass


class Sensor(metaclass=abc.ABCMeta):
    """ Sensor superclass. """
    
    def __init__(self, config: Dict[str, Any]):
        # Create configuration and override values
        self.cfg = SensorCfg()
        self.cfg.update(**config)

    @abc.abstractmethod
    def update(self) -> None:
        """ Update sensors state. """
        raise NotImplementedError('Must be implemented in subclass.')
