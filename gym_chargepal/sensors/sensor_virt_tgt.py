""" This file defines the sensors target class. """
# global
import logging
from dataclasses import dataclass
from rigmopy import Quaternion, Vector3d

# local
from gym_chargepal.sensors.sensor import SensorCfg, Sensor
from gym_chargepal.worlds.world_reacher import WorldReacher

# mypy
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


@dataclass
class VirtTgtSensorCfg(SensorCfg):
    sensor_id: str = 'virt_tgt_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'


class VirtTgtSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, config: Dict[str, Any], world: WorldReacher):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: VirtTgtSensorCfg = VirtTgtSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.world = world
        # Intern sensors state
        self.sensor_state = self.world.X_arm2tgt

    def update(self) -> None:
        self.sensor_state = self.world.X_arm2tgt
        
    def get_pos(self) -> Vector3d:
        return self.sensor_state.p

    def get_ori(self) -> Quaternion:
        return self.sensor_state.q
