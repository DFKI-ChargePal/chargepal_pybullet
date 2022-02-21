""" This file defines the virtual target sensor class. """
import logging
import copy
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, List, Optional
from chargepal_pybullet.rl_gym.worlds.world_pih import WorldPegInHole

from chargepal_pybullet.rl_gym.sensors.sensor import Sensor
from chargepal_pybullet.rl_gym.sensors.config import VIRTUAL_TARGET_SENSOR


LOGGER = logging.getLogger(__name__)


class VirtualTargetSensor(Sensor):
    """ Sensor of the virtual target frames. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config = copy.deepcopy(VIRTUAL_TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self._physics_client_id = world.physics_client_id
        self._target_id = world.pillar_id
        self._target_virtual_frame_idx = world.target_virtual_frame_idx
        self._link_state_idx = world.link_state_idx
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[Tuple[float, ...], ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = p.getLinkStates(
            self._target_id, self._target_virtual_frame_idx, True, True, physicsClientId=self._physics_client_id
        )

    def get_pos_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._link_state_idx.WORLD_LINK_FRAME_POS
        pos_list = [state[idx] for state in self._sensor_state]
        return pos_list

    def get_vel_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        idx = self._link_state_idx.WORLD_LINK_LIN_VEL
        pos_list: List[Tuple[float, ...]] = [state[idx] for state in self._sensor_state]
        return pos_list
