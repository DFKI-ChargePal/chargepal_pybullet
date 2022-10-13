""" This file defines the sensors tool class. """
import logging
import copy

# mypy
from typing import Dict, Any, Tuple, Union, Optional
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.worlds.world_tdt import WorldTopDownTask

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import PLUG_SENSOR
from gym_chargepal.bullet.config import BulletLinkState


LOGGER = logging.getLogger(__name__)


class PlugSensor(Sensor):
    """ Sensor of the arm plug. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole, WorldTopDownTask]):
        config = copy.deepcopy(PLUG_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self.world = world
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = self.world.bullet_client.getLinkState(
            bodyUniqueId=self.world.robot_id,
            linkIndex=self.world.plug_reference_frame_idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True
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

    def get_lin_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        vel = self._sensor_state[state_idx]
        return vel

    def get_ang_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_ANG_VEL
        vel = self._sensor_state[state_idx]
        return vel
