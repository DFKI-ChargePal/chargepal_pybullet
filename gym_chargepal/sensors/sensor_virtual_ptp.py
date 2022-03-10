""" This file defines the virtual target sensor class. """
import logging
import copy
import numpy as np
import pybullet as p

# mypy
from typing import Dict, Any, Tuple, List, Optional
from gym_chargepal.worlds.world_ptp import WorldPoint2Point

from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import VIRTUAL_TARGET_SENSOR
from gym_chargepal.bullet.bullet_observer import BulletObserver


LOGGER = logging.getLogger(__name__)


class VirtualTargetSensor(Sensor, BulletObserver):
    """ Sensor of the virtual target frames. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPoint2Point):
        config = copy.deepcopy(VIRTUAL_TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        BulletObserver.__init__(self)
        # params
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        assert self._world.target_pos
        assert self._world.target_ori
        
        pos_w = self._world.target_pos
        ori_w = self._world.target_ori

        q_zero = p.getQuaternionFromEuler((0.0, 0.0, 0.0), physicsClientId=self._physics_client_id)

        vrt_tgt_x, _ = p.multiplyTransforms(pos_w, ori_w, (1.0, 0.0, 0.0), q_zero, physicsClientId=self._physics_client_id)
        vrt_tgt_y, _ = p.multiplyTransforms(pos_w, ori_w, (0.0, 1.0, 0.0), q_zero, physicsClientId=self._physics_client_id)
        vrt_tgt_z, _ = p.multiplyTransforms(pos_w, ori_w, (0.0, 0.0, 1.0), q_zero, physicsClientId=self._physics_client_id)

        vec_zero = (0.0, 0.0, 0.0)
        self._sensor_state = (vrt_tgt_x, vrt_tgt_y, vrt_tgt_z, vec_zero, vec_zero, vec_zero)

    def get_pos_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        pos_list = list(self._sensor_state[0:3])
        return pos_list

    def get_vel_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        vel_list = list(self._sensor_state[3:6])
        return vel_list

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
