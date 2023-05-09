""" This file defines the sensors target class. """
from __future__ import annotations

# global
import logging
from dataclasses import dataclass
from rigmopy import Pose, Quaternion, Vector3d

# local
from gym_chargepal.sensors.sensor import SensorCfg, Sensor
from gym_chargepal.worlds.world_reacher import WorldReacher

# mypy
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class VirtTgtSensorCfg(SensorCfg):
    sensor_id: str = 'virt_tgt_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'


class VirtTgtSensor(Sensor):
    """ Sensor of the target frame. """
    def __init__(self, config: dict[str, Any], world: WorldReacher) -> None:
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: VirtTgtSensorCfg = VirtTgtSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.world = world
        # Intern sensors state
        self.sensor_state = self.world.vrt_tgt.X_arm2tgt

    def update(self) -> None:
        self.sensor_state = self.world.vrt_tgt.X_arm2tgt
    
    @property
    def noisy_X_arm2tgt(self) -> Pose:
        # TODO: Add signal noise
        return self.sensor_state

    @property
    def noisy_p_arm2tgt(self) -> Vector3d:
        return self.sensor_state.p

    @property
    def noisy_q_arm2tgt(self) -> Quaternion:
        return self.sensor_state.q
