# global
import logging
import pybullet as p
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Any, Dict, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class JointPositionMotorControlCfg(ConfigHandler):
    control_mode: int = p.POSITION_CONTROL
    target_vel: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    forces: Tuple[float, ...] = (330.0, 330.0, 150.0, 54.0, 54.0, 54.0)
    pos_gains: Tuple[float, ...] = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    vel_gains: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


class JointPositionMotorControl:
    """ Interface to use PyBullet Position Controller. """
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Create configuration object and update values
        self.cfg = JointPositionMotorControlCfg()
        self.cfg.update(**config)
        # Save references
        self.ur_arm = ur_arm

    def update(self, tgt_pos: Tuple[float, ...]) -> None:
        """ Update position controllers. Inputs are the desired joint angels of the arm. """
        self.ur_arm.bc.setJointMotorControlArray(
            bodyIndex=self.ur_arm.body_id,
            jointIndices=[idx for idx in self.ur_arm.joint_idx_dict.values()],
            controlMode=self.cfg.control_mode,
            targetPositions=tgt_pos,
            targetVelocities=self.cfg.target_vel,
            forces=self.cfg.forces,
            positionGains=self.cfg.pos_gains,
            velocityGains=self.cfg.vel_gains
        )
