""" This file defines the force torque sensors class. """
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
class FTSensorCfg(SensorCfg):
    sensor_id: str = 'ft_sensor'
    force_id: str = 'f'
    moment_id: str = 'm'
    overload: Tuple[float, ...] = (2000.0, 2000.0, 4000.0, 30.0, 30.0, 30.0)


class FTSensor(Sensor):
    """ Force Torque Sensor. """
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: FTSensorCfg = FTSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.ur_arm = ur_arm

    def meas_wrench(self) -> Tuple[float, ...]:
        # Mypy check whether ft sensor object exist 
        assert self.ur_arm.fts
        # Get sensor state and bring values in a range between -1.0 and +1.0
        meas = tuple([m/o for m, o in zip(self.ur_arm.fts.get_wrench(), self.cfg.overload)])
        # TODO: Add sensor noise
        return meas
