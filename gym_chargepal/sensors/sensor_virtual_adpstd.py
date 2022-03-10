""" This file defines the virtual target sensor class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, List, Optional
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import VIRTUAL_ADAPTER_STATION_SENSOR
from gym_chargepal.bullet.bullet_observer import BulletObserver
from gym_chargepal.bullet.config import BulletLinkState


LOGGER = logging.getLogger(__name__)


class VirtualAdapterStationSensor(Sensor, BulletObserver):
    """ Sensor of the virtual target frames. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config = copy.deepcopy(VIRTUAL_ADAPTER_STATION_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        BulletObserver.__init__(self)
        # params
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._adpstd_id = -1
        self._adpstd_ref_frame_idx: List[int] = []
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[Tuple[float, ...], ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getLinkStates(
            bodyUniqueId=self._adpstd_id,
            linkIndex=self._adpstd_ref_frame_idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
            physicsClientId=self._physics_client_id
        )

    def get_pos_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        pos_list = [state[state_idx] for state in self._sensor_state]
        return pos_list

    def get_vel_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        pos_list: List[Tuple[float, ...]] = [state[state_idx] for state in self._sensor_state]
        return pos_list

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._adpstd_id = self._world.adpstd_id
        self._adpstd_ref_frame_idx = [idx for idx in self._world.adpstd_ref_frame_idx_dict.values()]
