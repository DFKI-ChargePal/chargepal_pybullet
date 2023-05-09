from __future__ import annotations
# global
import logging
import numpy as np
from dataclasses import dataclass
from rigmopy import Pose, Quaternion, Vector3d

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


class VirtualTargetCfg(ConfigHandler):
    X_arm2tgt: Pose = Pose().from_xyz((0.0, 0.0, 1.2)).from_euler_angle(angles=(np.pi/2, 0.0, 0.0))


class VirtualTarget:

    def __init__(self, config: dict[str, Any]) -> None:
        # Create configuration and overwrite values
        self.cfg = VirtualTargetCfg()
        self.cfg.update(**config)

    @property
    def X_arm2tgt(self) -> Pose:
        return self.cfg.X_arm2tgt
    
    @property
    def p_arm2tgt(self) -> Vector3d:
        return self.X_arm2tgt.p
    
    @property
    def q_arm2tgt(self) -> Quaternion:
        return self.X_arm2tgt.q
