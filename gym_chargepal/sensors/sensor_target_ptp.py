""" This file defines the sensors target class. """
# global
import copy
import logging

# local
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import TARGET_SENSOR

# mypy
from typing import Dict, Any, Tuple
from gym_chargepal.worlds.world_ptp import WorldPoint2Point


LOGGER = logging.getLogger(__name__)


class TargetSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPoint2Point):
        config = copy.deepcopy(TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self.world = world
        # intern sensors state
        self.sensor_state = (self.world.target_pos, self.world.target_ori)

    def update(self) -> None:
        self.sensor_state = (self.world.target_pos, self.world.target_ori)

    def get_pos(self) -> Tuple[float, ...]:
        return self.sensor_state[0]

    def get_ori(self) -> Tuple[float, ...]:
        return self.sensor_state[1]
