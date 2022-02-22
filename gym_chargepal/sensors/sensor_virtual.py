""" This file defines the virtual sensors class. """
import logging
import copy

# mypy
from typing import Dict, Any, Tuple

from chargepal_pybullet.gym_chargepal.sensors.sensor import Sensor
from chargepal_pybullet.gym_chargepal.sensors.config import VIRTUAL_SENSOR


LOGGER = logging.getLogger(__name__)


class VirtualSensor(Sensor):
    """ Virtual Sensor. """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(VIRTUAL_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # intern sensors state
        self._sensor_state: Tuple[float, ...] = self._hyperparams['state']

    def update(self) -> None:
        pass

    def measurement(self) -> Tuple[float, ...]:
        return self._sensor_state
