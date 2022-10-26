""" This file defines the sensors tool class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Dict, Any, Tuple, Union, Optional


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
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: PlugSensorCfg = PlugSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.ur_arm = ur_arm

    def get_pos(self) -> Tuple[float, ...]:
        return self.ur_arm.tcp.get_pos()

    def get_ori(self) -> Tuple[float, ...]:
        return self.ur_arm.tcp.get_ori()

    def get_lin_vel(self) -> Tuple[float, ...]:
        return self.ur_arm.tcp.get_lin_vel()

    def get_ang_vel(self) -> Tuple[float, ...]:
        return self.ur_arm.tcp.get_ang_vel()
