""" This file defines the force torque sensors class. """
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
from typing import Dict, Any, Tuple
from numpy import typing as npt


LOGGER = logging.getLogger(__name__)


@dataclass
class FTSensorCfg(SensorCfg):
    sensor_id: str = 'ft_sensor'
    force_id: str = 'f'
    moment_id: str = 'm'
    ft_range: Tuple[float, ...] = (500.0, 500.0, 1200.0, 15.0, 15.0, 12.0)
    overload: Tuple[float, ...] = (2000.0, 2000.0, 4000.0, 30.0, 30.0, 30.0)
    render_bar: bool = False
    render_bar_length: int = 35
    color_bound_med: float = 0.7
    color_bound_high: float = 0.95


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
        # Convert limits to numpy arrays
        self.ft_min = -np.array(self.cfg.ft_range, dtype=np.float32)
        self.ft_max = np.array(self.cfg.ft_range, dtype=np.float32)

    @property
    def wrench(self) -> Vector6d:
        # Mypy check whether ft sensor object exist 
        assert self.ur_arm._fts
        # Get sensor state and bring values in a range between -1.0 and +1.0
        wrench = self.ur_arm._fts.get_wrench()
        norm_wrench: npt.NDArray[np.float_] = np.clip(wrench.to_numpy(), self.ft_min, self.ft_max) / self.ft_max
        return Vector6d().from_xyzXYZ(norm_wrench)

    @property
    def noisy_wrench(self) -> Vector6d:
        meas = self.wrench
        # TODO: Add sensor noise
        return meas

    def render_ft_bar(self, render: bool) -> None:
        if render and self.cfg.render_bar:
            wrench = self.wrench.xyzXYZ
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
