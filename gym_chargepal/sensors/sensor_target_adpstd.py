""" This file defines the sensors target class. """
# global
import copy
import logging

# local
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import TARGET_SENSOR
from gym_chargepal.bullet.config import BulletLinkState

# mypy
from typing import Dict, Any, Tuple, Optional
from gym_chargepal.worlds.world_pih import WorldPegInHole


LOGGER = logging.getLogger(__name__)


class TargetSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config = copy.deepcopy(TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self.world = world
        # intern sensors state
        self.sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self.sensor_state = self.world.bullet_client.getLinkState(
            bodyUniqueId=self.world.adpstd_id,
            linkIndex=self.world.adpstd_reference_frame_idx,
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
