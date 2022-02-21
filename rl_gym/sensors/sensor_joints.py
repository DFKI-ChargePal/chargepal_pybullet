""" This file defines the sensors joint class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, Union, Optional
from chargepal_pybullet.rl_gym.worlds.world_ptp import WorldPoint2Point
from chargepal_pybullet.rl_gym.worlds.world_pih import WorldPegInHole

from chargepal_pybullet.rl_gym.sensors.sensor import Sensor
from chargepal_pybullet.rl_gym.sensors.config import JOINT_SENSOR


LOGGER = logging.getLogger(__name__)


class JointSensor(Sensor):
    """ Sensor of the arm joints. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(JOINT_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self._physics_client_id = world.physics_client_id
        self._arm_id = world.arm_id
        self._joint_idx = world.joint_idx
        self._joint_idx_list = [idx for idx in world.joint_idx.values()]
        self._joint_state_idx = world.joint_state_idx
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getJointStates(
            self._arm_id, self._joint_idx_list, physicsClientId=self._physics_client_id
        )

    def get_pos(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._joint_state_idx.JOINT_POSITION
        pos = tuple(joint[idx] for joint in self._sensor_state)
        return pos

    def get_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._joint_state_idx.JOINT_VELOCITY
        vel = tuple(joint[idx] for joint in self._sensor_state)
        return vel
