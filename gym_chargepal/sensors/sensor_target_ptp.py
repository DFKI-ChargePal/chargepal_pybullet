""" This file defines the sensors target class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple
from gym_chargepal.worlds.world_ptp import WorldPoint2Point

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import TARGET_SENSOR


LOGGER = logging.getLogger(__name__)


class TargetSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPoint2Point):
        config = copy.deepcopy(TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self._world = world
        # intern sensors state
        self._sensor_state = (self._world.target_pos, self._world.target_ori)

    def update(self) -> None:
        self._sensor_state = (self._world.target_pos, self._world.target_ori)

    def get_pos(self) -> Tuple[float, ...]:
        return self._sensor_state[0]

    def get_ori(self) -> Tuple[float, ...]:
        return self._sensor_state[1]