""" This file defines the socket sensor class. """
# global
import logging
import copy
import pybullet as p

# local
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import SOCKET_SENSOR
from gym_chargepal.bullet.config import BulletLinkState
from gym_chargepal.bullet.bullet_observer import BulletObserver


# mypy
from typing import Any, Dict, Optional, Tuple
from gym_chargepal.worlds.world_tdt import WorldTopDownTask


LOGGER = logging.getLogger(__name__)


class SocketSensor(Sensor, BulletObserver):
    """ Sensor class for a fake socket observation. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldTopDownTask):
        config = copy.deepcopy(SOCKET_SENSOR)
        config.update(hyperparams)
        super().__init__(config)
        # params
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._socket_id = -1
        self._socket_ref_frame_idx = -1
        # intern sensor state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getLinkState(
            bodyUniqueId=self._socket_id,
            linkIndex=self._socket_ref_frame_idx,
            computeLinkVelocity=False,
            computeForwardKinematics=False,
            physicsClientId=self._physics_client_id
        )

    def get_pos(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        pos = self._sensor_state[state_idx]
        return pos

    def get_ori(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        ori = self._sensor_state[state_idx]
        return ori

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._socket_id = self._world.socket_id
        self._socket_ref_frame_idx = self._world.socket_frame_idx
