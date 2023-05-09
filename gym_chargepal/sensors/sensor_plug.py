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
    def X_arm2sensor(self) -> Pose:
        return self.ur_arm.get_X_base2tcp()

    @property
    def p_arm2sensor(self) -> Vector3d:
        return self.ur_arm.get_p_base2tcp()

    @property
    def q_arm2sensor(self) -> Quaternion:
        return self.ur_arm.get_q_base2tcp()

    @property
    def twist(self) -> Vector6d:
        return self.ur_arm.tcp_link.get_twist_world2link()
