""" This file defines the virtual target sensor class. """
# global
import copy
import logging

# local
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import VIRTUAL_TARGET_SENSOR

# mypy
from typing import Dict, Any, Tuple, List, Optional
from gym_chargepal.worlds.world_ptp import WorldPoint2Point


LOGGER = logging.getLogger(__name__)


class VirtualTargetSensor(Sensor):
    """ Sensor of the virtual target frames. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPoint2Point):
        config = copy.deepcopy(VIRTUAL_TARGET_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self.world = world
        # intern sensors state
        self.sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        assert self.world.target_pos
        assert self.world.target_ori
        
        pos_w = self.world.target_pos
        ori_w = self.world.target_ori

        q_zero = self.world.bullet_client.getQuaternionFromEuler((0.0, 0.0, 0.0))

        vrt_tgt_x, _ = self.world.bullet_client.multiplyTransforms(pos_w, ori_w, (1.0, 0.0, 0.0), q_zero)
        vrt_tgt_y, _ = self.world.bullet_client.multiplyTransforms(pos_w, ori_w, (0.0, 1.0, 0.0), q_zero)
        vrt_tgt_z, _ = self.world.bullet_client.multiplyTransforms(pos_w, ori_w, (0.0, 0.0, 1.0), q_zero)

        vec_zero = (0.0, 0.0, 0.0)
        self.sensor_state = (vrt_tgt_x, vrt_tgt_y, vrt_tgt_z, vec_zero, vec_zero, vec_zero)

    def get_pos_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        pos_list = list(self.sensor_state[0:3])
        return pos_list

    def get_vel_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        vel_list = list(self.sensor_state[3:6])
        return vel_list
