""" This file defines the force torque sensors class. """
import logging
import copy

# mypy
from typing import Dict, Any, Tuple, Optional
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import FT_SENSOR
from gym_chargepal.bullet.config import BulletJointState

LOGGER = logging.getLogger(__name__)


class FTSensor(Sensor):
    """ Force Torque Sensor. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config: Dict[str, Any] = copy.deepcopy(FT_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self.world = world
        # intern sensors state
        self.sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self.sensor_state = self.world.bullet_client.getJointState(
            bodyUniqueId=self.world.bullet_client,
            jointIndex=self.world.ft_sensor_joint_idx
        )

    def measurement(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        # ToDo: There shout be a noisy measurement signal in future versions.
        state_idx = BulletJointState.JOINT_REACTION_FORCE
        meas = self.sensor_state[state_idx]
        return meas
