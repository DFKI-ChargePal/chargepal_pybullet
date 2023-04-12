""" This file defines the socket sensor class. """
# global
import logging
import numpy as np
from dataclasses import dataclass
from rigmopy import Quaternion, Vector3d

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


    def get_pos(self) -> Vector3d:
        return self.socket.socket.get_pos()

    def get_ori(self) -> Quaternion:
        return self.socket.socket.get_ori()

    def meas_pos(self) -> Vector3d:
        gt_pos = self.get_pos().xyz
        pos_meas: npt.NDArray[np.float64] = np.array(gt_pos[0:3], dtype=np.float64) + np.random.randn(3) * self.pos_noise + self.pos_bias
        return Vector3d().from_xyz(pos_meas)

    def meas_ori(self) -> Quaternion:
        gt_ori_eul = np.array(self.get_ori().to_euler_angle(), dtype=np.float64)
        ori_eul_meas: npt.NDArray[np.float64] = gt_ori_eul + np.random.randn(3) * self.ori_noise + self.ori_bias
        ori_meas = Quaternion().from_xyzw(self.socket.bc.getQuaternionFromEuler(ori_eul_meas.tolist()))
        return ori_meas
