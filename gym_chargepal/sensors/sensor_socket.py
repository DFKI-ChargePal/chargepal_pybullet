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
    var_lin: tuple[float, ...] = (0.0, 0.0, 0.0)  # linear position sensor noise
    var_ang: tuple[float, ...] = (0.0, 0.0, 0.0)  # angular (euler) sensor noise


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

    @property
    def noisy_X_arm2sensor(self) -> Pose:
        
        if bool(np.random.randint(2)):
            # Sample in negative space
            random_pos = Vector3d().from_xyz(np.random.uniform((0.0, -0.025, -0.025), (0.0, -0.020, -0.020)))
        else:
            # Sample in positive space
            random_pos = Vector3d().from_xyz(np.random.uniform((0.0, 0.020, 0.020), (0.0, 0.025, 0.025)))
        
        noisy_X = self.socket.X_arm2socket.random((0.0, 0.0, 0.0), self.cfg.var_ang)
        noisy_X = noisy_X.from_pq(p=noisy_X.p + random_pos)
        # noisy_X = self.socket.X_arm2socket.random(self.cfg.var_lin, self.cfg.var_ang)
        return noisy_X

    @property
    def noisy_p_arm2sensor(self) -> Vector3d:
        return self.noisy_X_arm2sensor.p

    @property
    def noisy_q_arm2sensor(self) -> Quaternion:
        return self.noisy_X_arm2sensor.q
