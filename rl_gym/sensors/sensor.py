""" This file defines the sensors base class. """
import abc
import copy

# mypy
from typing import Dict, Any

from gym_env.sensors.config import SENSOR


class Sensor(object):
    """ Sensor superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(SENSOR)
        config.update(hyperparams)
        self._hyperparams = config

    @abc.abstractmethod
    def update(self) -> None:
        """ Update sensors state. """
        raise NotImplementedError('Must be implemented in subclass.')
