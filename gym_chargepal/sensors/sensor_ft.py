""" This file defines the force torque sensors class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, Optional
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import FT_SENSOR
from gym_chargepal.bullet.bullet_observer import BulletObserver
from gym_chargepal.bullet.config import BulletJointState

LOGGER = logging.getLogger(__name__)


class FTSensor(Sensor, BulletObserver):
    """ Force Torque Sensor. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config: Dict[str, Any] = copy.deepcopy(FT_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        BulletObserver.__init__(self)
        # params
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._robot_id = -1
        self._ft_sensor_joint_idx = -1
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getJointState(
            bodyUniqueId=self._robot_id,
            jointIndex=self._ft_sensor_joint_idx,
            physicsClientId=self._physics_client_id
        )

    def measurement(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        # ToDo: There shout be a noisy measurement signal in future versions.
        state_idx = BulletJointState.JOINT_REACTION_FORCE
        meas = self._sensor_state[state_idx]
        return meas

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._robot_id = self._world.robot_id
        self._ft_sensor_joint_idx = self._world.ft_sensor_joint_idx
