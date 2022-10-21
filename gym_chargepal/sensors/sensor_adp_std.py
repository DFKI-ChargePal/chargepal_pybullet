""" This file defines the sensors target class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.config import BulletLinkState
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Dict, Any, Tuple, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class AdpStdSensorCfg(SensorCfg):
    sensor_id: str = 'adp_std_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'


class AdpStdSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, config: Dict[str, Any], world: WorldPegInHole):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: AdpStdSensorCfg = AdpStdSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.world = world
        # intern sensors state
        self.sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self.sensor_state = self.world.bullet_client.getLinkState(
            bodyUniqueId=self.world.adp_std_id,
            linkIndex=self.world.adp_std_reference_frame_idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True
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
