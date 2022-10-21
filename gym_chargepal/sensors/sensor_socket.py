""" This file defines the socket sensor class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.config import BulletLinkState
from gym_chargepal.sensors.sensor import SensorCfg, Sensor
from gym_chargepal.worlds.world_tdt import WorldTopDownTask

# mypy
from typing import Any, Dict, Optional, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class SocketSensorCfg(SensorCfg):
    sensor_id: str = 'socket_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'


class SocketSensor(Sensor):
    """ Sensor class for a fake socket observation. """
    def __init__(self, config: Dict[str, Any], world: WorldTopDownTask):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: SocketSensorCfg = SocketSensorCfg()
        self.cfg.update(**config)
        # Safe references
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

