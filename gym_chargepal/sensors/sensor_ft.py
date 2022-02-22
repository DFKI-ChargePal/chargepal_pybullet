""" This file defines the force torque sensors class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, Optional
from chargepal_pybullet.gym_chargepal.worlds.world_pih import WorldPegInHole

from chargepal_pybullet.gym_chargepal.sensors.sensor import Sensor
from chargepal_pybullet.gym_chargepal.sensors.config import FT_SENSOR


LOGGER = logging.getLogger(__name__)


class FTSensor(Sensor):
    """ Force Torque Sensor. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config: Dict[str, Any] = copy.deepcopy(FT_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self._physics_client_id = world.physics_client_id
        self._arm_id = world.arm_id
        self._ft_sensor_idx = world.ft_sensor_idx
        self._joint_state_idx = world.joint_state_idx
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getJointState(
            self._arm_id, self._ft_sensor_idx, physicsClientId=self._physics_client_id
        )

    def measurement(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        # ToDo: There shout be a noisy measurement signal in future versions.
        idx = self._joint_state_idx.JOINT_REACTION_FORCE
        meas = self._sensor_state[idx]
        return meas
