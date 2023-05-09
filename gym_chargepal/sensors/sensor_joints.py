""" This file defines the sensors joint class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Dict, Any, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class JointSensorCfg(SensorCfg):
    sensor_id: str = 'joint_sensor'
    pos_id: str = 'joint_pos'
    vel_id: str = 'joint_vel'
    acc_id: str = 'joint_acc'


class JointSensor(Sensor):
    """ Sensor of the arm joints. """
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: JointSensorCfg = JointSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.ur_arm = ur_arm

    @property
    def pos(self) -> Tuple[float, ...]:
        return self.ur_arm.get_joint_pos()

    @property
    def vel(self) -> Tuple[float, ...]:
        return self.ur_arm.get_joint_vel()
