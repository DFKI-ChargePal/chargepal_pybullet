""" This file defines the socket sensor class. """
# global
import copy
import logging

# local
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import SOCKET_SENSOR
from gym_chargepal.bullet.config import BulletLinkState

# mypy
from typing import Any, Dict, Optional, Tuple
from gym_chargepal.worlds.world_tdt import WorldTopDownTask


LOGGER = logging.getLogger(__name__)


class SocketSensor(Sensor):
    """ Sensor class for a fake socket observation. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldTopDownTask):
        config = copy.deepcopy(SOCKET_SENSOR)
        config.update(hyperparams)
        super().__init__(config)
        # params
        self.world = world
        # intern sensor state
        self.sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self.sensor_state = self.world.bullet_client.getLinkState(
            bodyUniqueId=self.world.socket_id,
            linkIndex=self.world.socket_frame_idx,
            computeLinkVelocity=False,
            computeForwardKinematics=False
        )

    def get_pos(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        pos = self.sensor_state[state_idx]
        return pos

    def get_ori(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        ori = self.sensor_state[state_idx]
        return ori

