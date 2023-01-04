""" This file defines the socket sensor class. """
# global
import logging
import numpy as np
from dataclasses import dataclass
from rigmopy import Orientation, Position

# local
from gym_chargepal.bullet.socket import Socket
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from numpy import typing as npt
from typing import Any, Dict, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class SocketSensorCfg(SensorCfg):
    sensor_id: str = 'socket_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'
    pos_noise: Tuple[float, ...] = (0.0, 0.0, 0.0)  # linear position sensor noise
    pos_bias: Tuple[float, ...] = (0.0, 0.0, 0.0)   # linear position sensor bias
    ori_noise: Tuple[float, ...] = (0.0, 0.0, 0.0)  # angular (euler) sensor noise 
    ori_bias: Tuple[float, ...] = (0.0, 0.0, 0.0)   # angular (euler) sensor bias


class SocketSensor(Sensor):
    """ Sensor class for a fake socket observation. """
    def __init__(self, config: Dict[str, Any], socket: Socket):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: SocketSensorCfg = SocketSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.socket = socket
        self.pos_noise = np.array(self.cfg.pos_noise, dtype=np.float32)
        self.pos_bias = np.array(self.cfg.pos_bias, dtype=np.float32)
        self.ori_noise = np.array(self.cfg.ori_noise, dtype=np.float32)
        self.ori_bias = np.array(self.cfg.ori_bias, dtype=np.float32)


    def get_pos(self) -> Position:
        return self.socket.socket.get_pos()

    def get_ori(self) -> Orientation:
        return self.socket.socket.get_ori()

    def meas_pos(self) -> Position:
        gt_pos = self.get_pos().as_np_vec()
        pos_meas: npt.NDArray[np.float32] = gt_pos + np.random.randn(3) * self.pos_noise + self.pos_bias
        return Position().from_vec(pos_meas.tolist())

    def meas_ori(self) -> Orientation:
        gt_ori = self.get_ori().as_vec(order='xyzw')
        gt_ori_eul = np.array(self.socket.bc.getEulerFromQuaternion(gt_ori), dtype=np.float32)
        ori_eul_meas: npt.NDArray[np.float32] = gt_ori_eul + np.random.randn(3) * self.ori_noise + self.ori_bias
        ori_meas = Orientation().from_vec(self.socket.bc.getQuaternionFromEuler(ori_eul_meas.tolist()), order='xyzw')
        return ori_meas
