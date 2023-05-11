""" This file defines the force torque sensors class. """
from __future__ import annotations

# global
import logging
import numpy as np
from rigmopy import Vector6d
from termcolor import cprint
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class FTSensorCfg(SensorCfg):
    sensor_id: str = 'ft_sensor'
    force_id: str = 'f'
    moment_id: str = 'm'
    render_bar: bool = False
    render_bar_length: int = 35
    color_bound_med: float = 0.7
    color_bound_high: float = 0.95


class FTSensor(Sensor):
    """ Force Torque Sensor. """
    def __init__(self, config: dict[str, Any], ur_arm: URArm):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: FTSensorCfg = FTSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.ur_arm = ur_arm

    @property
    def noisy_wrench(self) -> Vector6d:
        meas = self.ur_arm.wrench
        # TODO: Add sensor noise
        return meas

    def render_ft_bar(self, render: bool) -> None:
        if render and self.cfg.render_bar:
            wrench = self.ur_arm.wrench.xyzXYZ
            # Find outliers
            ft_max = max(wrench)
            # idx = wrench.index(ft_max)
            # Scale to bar and build command line string
            n_char = round(self.cfg.render_bar_length * ft_max)
            bar_str = '|' + n_char * '#' + (self.cfg.render_bar_length - n_char) * '.' + '|'
            if ft_max <= self.cfg.color_bound_med:
                color = 'green'
            elif ft_max <= self.cfg.color_bound_high:
                color = 'yellow'
            else:
                color = 'red'
            cprint(bar_str, color=color)
