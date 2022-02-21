""" This file defines the sensors tool class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, Union, Optional
from chargepal_pybullet.rl_gym.worlds.world_ptp import WorldPoint2Point
from chargepal_pybullet.rl_gym.worlds.world_pih import WorldPegInHole

from chargepal_pybullet.rl_gym.sensors.sensor import Sensor
from chargepal_pybullet.rl_gym.sensors.config import TOOL_SENSOR


LOGGER = logging.getLogger(__name__)


class ToolSensor(Sensor):
    """ Sensor of the arm tool (plug). """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config = copy.deepcopy(TOOL_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self._physics_client_id = world.physics_client_id
        self._arm_id = world.arm_id
        self._tool_frame_idx = world.tool_frame_idx
        self._link_state_idx = world.link_state_idx
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getLinkState(
            self._arm_id, self._tool_frame_idx, True, True, physicsClientId=self._physics_client_id
        )

    def get_pos(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._link_state_idx.WORLD_LINK_FRAME_POS
        pos = self._sensor_state[idx]
        return pos

    def get_ori(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._link_state_idx.WORLD_LINK_FRAME_ORI
        ori = self._sensor_state[idx]
        return ori

    def get_lin_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._link_state_idx.WORLD_LINK_LIN_VEL
        vel = self._sensor_state[idx]
        return vel

    def get_ang_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._link_state_idx.WORLD_LINK_ANG_VEL
        vel = self._sensor_state[idx]
        return vel
