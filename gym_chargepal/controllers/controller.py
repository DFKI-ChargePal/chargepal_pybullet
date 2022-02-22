""" This file defines the controllers base class. """
import abc
import copy

# mypy
from typing import Dict, Any, Generic, TypeVar

from chargepal_pybullet.gym_chargepal.controllers.config import CONTROLLER


class Controller(object):
    """ Controller superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(CONTROLLER)
        config.update(hyperparams)
        self._hyperparams = config
