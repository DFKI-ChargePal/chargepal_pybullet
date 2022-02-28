""" This file defines the sensors joint class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, List, Tuple, Union, Optional
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import JOINT_SENSOR
from gym_chargepal.bullet.bullet_observer import BulletObserver
from gym_chargepal.bullet.config import BulletJointState


LOGGER = logging.getLogger(__name__)


class JointSensor(Sensor, BulletObserver):
    """ Sensor of the arm joints. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(JOINT_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        BulletObserver.__init__(self)
        # params
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._robot_id = -1
        self._ur_joint_idx: List[int] = []
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getJointStates(
            bodyUniqueId=self._robot_id,
            jointIndices=self._ur_joint_idx,
            physicsClientId=self._physics_client_id
        )

    def get_pos(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletJointState.JOINT_POSITION
        pos = tuple(joint[state_idx] for joint in self._sensor_state)
        return pos

    def get_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletJointState.JOINT_VELOCITY
        vel = tuple(joint[state_idx] for joint in self._sensor_state)
        return vel

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._robot_id = self._world.robot_id
        self._ur_joint_idx = [idx for idx in self._world.ur_joint_idx_dict.values()]
