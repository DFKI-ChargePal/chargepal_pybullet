""" This file defines the sensors target class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, Optional
from gym_env.worlds.world_pih import WorldPegInHole

from gym_env.sensors.sensor import Sensor
from gym_env.sensors.config import TARGET_SENSOR


LOGGER = logging.getLogger(__name__)


class TargetSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config = copy.deepcopy(TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self._physics_client_id = world.physics_client_id
        self._target_id = world.pillar_id
        self._target_frame_idx = world.target_frame_idx
        self._link_state_idx = world.link_state_idx
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getLinkState(
            self._target_id, self._target_frame_idx, True, True, physicsClientId=self._physics_client_id
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
