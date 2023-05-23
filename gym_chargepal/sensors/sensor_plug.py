""" This file defines the sensors tool class. """
from __future__ import annotations

# global
import logging
from dataclasses import dataclass
from rigmopy import Pose, Quaternion, Vector3d, Vector6d

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class PlugSensorCfg(SensorCfg):
    sensor_id: str = 'plug_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'
    lin_vel_id: str = 'v'
    ang_vel_id: str = 'w'


class PlugSensor(Sensor):
    """ Sensor of the arm plug. """
    def __init__(self, config: dict[str, Any], ur_arm: URArm):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: PlugSensorCfg = PlugSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.ur_arm = ur_arm
    
    @property
    def noisy_X_arm2sensor(self) -> Pose:
        # TODO: Add noise
        return self.ur_arm.X_arm2plug

    @property
    def noisy_p_arm2sensor(self) -> Vector3d:
        return self.ur_arm.p_arm2plug

    @property
    def noisy_q_arm2sensor(self) -> Quaternion:
        return self.ur_arm.q_arm2plug

    @property
    def noisy_V_wrt_world(self) -> Vector6d:
        # TODO: Add noise
        return self.ur_arm.tcp_link.twist_world2link

    @property
    def noisy_V_wrt_arm(self) -> Vector6d:
        V_wrt_world = self.noisy_V_wrt_world
        v_wrt_world, w_wrt_world = V_wrt_world.split()
        v_wrt_arm = self.ur_arm.q_world2arm.apply(v_wrt_world)
        w_wrt_arm = self.ur_arm.q_world2arm.apply(w_wrt_world)
        return Vector6d().from_xyzXYZ(v_wrt_arm.xyz + w_wrt_arm.xyz)
