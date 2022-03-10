""" This file defines the sensors target class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, Optional
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import TARGET_SENSOR
from gym_chargepal.bullet.bullet_observer import BulletObserver
from gym_chargepal.bullet.config import BulletLinkState


LOGGER = logging.getLogger(__name__)


class TargetSensor(Sensor, BulletObserver):
    """ Sensor of the target frame. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config = copy.deepcopy(TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        BulletObserver.__init__(self)
        # params
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._adpstd_id = -1
        self._adpstd_ref_frame_idx = -1
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getLinkState(
            bodyUniqueId=self._adpstd_id,
            linkIndex=self._adpstd_ref_frame_idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
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
        self._adpstd_id = self._world.adpstd_id
        self._adpstd_ref_frame_idx = self._world.adpstd_reference_frame_idx
