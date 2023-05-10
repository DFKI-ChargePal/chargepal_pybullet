""" This file defines the socket sensor class. """
from __future__ import annotations

# global
import logging
import numpy as np
from dataclasses import dataclass
from rigmopy import Pose, Quaternion, Vector3d

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.socket import Socket
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Any
from numpy import typing as npt


LOGGER = logging.getLogger(__name__)


@dataclass
class SocketSensorCfg(SensorCfg):
    sensor_id: str = 'socket_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'
    pos_noise: tuple[float, ...] = (0.0, 0.0, 0.0)  # linear position sensor noise
    pos_bias: tuple[float, ...] = (0.0, 0.0, 0.0)   # linear position sensor bias
    ori_noise: tuple[float, ...] = (0.0, 0.0, 0.0)  # angular (euler) sensor noise 
    ori_bias: tuple[float, ...] = (0.0, 0.0, 0.0)   # angular (euler) sensor bias


class SocketSensor(Sensor):
    """ Sensor class for a fake socket observation. """
    def __init__(self, config: dict[str, Any], ur_arm: URArm, socket: Socket):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: SocketSensorCfg = SocketSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.ur_arm = ur_arm
        self.socket = socket
        # Set sensor noise
        self.pos_noise = np.array(self.cfg.pos_noise, dtype=np.float32)
        self.pos_bias = np.array(self.cfg.pos_bias, dtype=np.float32)
        self.ori_noise = np.array(self.cfg.ori_noise, dtype=np.float32)
        self.ori_bias = np.array(self.cfg.ori_bias, dtype=np.float32)

    @property
    def X_arm2sensor(self) -> Pose:
        X_world2socket = self.socket.socket.X_world2link
        X_world2arm = self.ur_arm.X_world2arm
        X_arm2socket = X_world2arm.inverse() * X_world2socket
        return X_arm2socket

    @property
    def p_arm2sensor(self) -> Vector3d:
        return self.X_arm2sensor.p

    @property
    def q_arm2sensor(self) -> Quaternion:
        return self.X_arm2sensor.q

    @property
    def noisy_p_arm2sensor(self) -> Vector3d:
        gt_pos = self.p_arm2sensor.xyz
        pos_meas: npt.NDArray[np.float64] = np.array(gt_pos[0:3], dtype=np.float64) + np.random.randn(3) * self.pos_noise + self.pos_bias
        return Vector3d().from_xyz(pos_meas)

    @property
    def noisy_q_arm2sensor(self) -> Quaternion:
        gt_ori_eul = np.array(self.q_arm2sensor.to_euler_angle(), dtype=np.float64)
        ori_eul_meas: npt.NDArray[np.float64] = gt_ori_eul + np.random.randn(3) * self.ori_noise + self.ori_bias
        ori_meas = Quaternion().from_xyzw(self.socket.bullet_client.getQuaternionFromEuler(ori_eul_meas.tolist()))
        return ori_meas
