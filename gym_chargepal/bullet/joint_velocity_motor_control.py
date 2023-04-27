# global
import logging
import pybullet as p
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Dict, Any, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class JointVelocityMotorControlCfg(ConfigHandler):
    control_mode: int = p.VELOCITY_CONTROL
    target_pos: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # no effect in velocity control mode
    forces: Tuple[float, ...] = (330.0, 330.0, 150.0, 54.0, 54.0, 54.0)
    pos_gains: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # no effect in velocity control mode
    vel_gains: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


class JointVelocityMotorControl:
    """ Interface to use PyBullet Velocity Controller. """
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Create configuration object and update values
        self.cfg = JointVelocityMotorControlCfg()
        self.cfg.update(**config)
        # Save references
        self.ur_arm = ur_arm

    def update(self, tgt_vel: Tuple[float, ...]) -> None:
        """ Update velocity controllers. Inputs are the desired joint velocities of the arm. """
        self.ur_arm.bullet_client.setJointMotorControlArray(
            bodyIndex=self.ur_arm._body_id,
            jointIndices=[idx for idx in self.ur_arm._joint_idx_dict.values()],
            controlMode=self.cfg.control_mode,
            targetPositions=self.cfg.target_pos,
            targetVelocities=tgt_vel,
            forces=self.cfg.forces,
            positionGains=self.cfg.pos_gains,
            velocityGains=self.cfg.vel_gains
        )
