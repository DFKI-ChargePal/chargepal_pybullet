from __future__ import annotations

# global
from rigmopy import Vector6d
from dataclasses import dataclass

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler
from gym_chargepal.utility.pd_controller import PDController

# typing
from typing import Any, Tuple

Tuple6DType = Tuple[float, float, float, float, float, float]


@dataclass
class SpatialPDControllerCfg(ConfigHandler):
    period: float = -1.0
    kp: Tuple6DType = (0.5, 0.5, 0.5, 0.1, 0.1, 0.1)
    kd: Tuple6DType = (0.001, 0.001, 0.001, 0.001, 0.001, 0.001)


class SpatialPDController:

    def __init__(self, config: dict[str, Any]) -> None:
        """ Six dimensional spatial PD controller

        Args:
            config: Dictionary to overwrite kp and kd values
        """
        # Create configuration and override values
        self.cfg = SpatialPDControllerCfg()
        self.cfg.update(**config)
        self.ctrl_l = [PDController(kp, kd) for kp, kd in zip(self.cfg.kp, self.cfg.kd)]
        if self.cfg.period < 0.0:
            raise ValueError(f"Controller period ({self.cfg.period}) smaller than 0.0."
                             f"Probably not set via config dictionary: {config}")

    def reset(self) -> None:
        """ Reset all sub-controllers """
        for ctrl in self.ctrl_l:
            ctrl.reset()

    def update(self, errors: Vector6d) -> Vector6d:
        """ Update sub-controllers

        Args:
            errors: Controller error list [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
            period: Controller update duration [sec]

        Returns:
            List with new controller outputs
        """
        out = [ctrl.update(err, self.cfg.period) for ctrl, err in zip(self.ctrl_l, errors.xyzXYZ)]
        return Vector6d().from_xyzXYZ(out)
