import copy
import logging

# mypy
from typing import Dict, Any, Tuple, List
from gym_chargepal.bullet.ur_arm import URArm

from gym_chargepal.bullet.config import JOINT_POSITION_MOTOR_CONTROL


LOGGER = logging.getLogger(__name__)


class JointPositionMotorControl:
    """ Interface to use PyBullet Position Controller. """
    def __init__(self, hyperparams: Dict[str, Any], ur_arm: URArm):
        config: Dict[str, Any] = copy.deepcopy(JOINT_POSITION_MOTOR_CONTROL)
        config.update(hyperparams)
        self.hyperparams = config
        # Save references
        self.ur_arm = ur_arm
        # constants
        self.control_mode: int = self.hyperparams['control_mode']
        self.target_vel: List[float] = self.hyperparams['target_vel']
        self.forces: List[float] = self.hyperparams['forces']
        self.pos_gains: List[float] = self.hyperparams['pos_gains']
        self.vel_gains: List[float] = self.hyperparams['vel_gains']

    def update(self, tgt_pos: Tuple[float, ...]) -> None:
        """ Update position controllers. Inputs are the desired joint angels of the arm. """
        self.ur_arm.bc.setJointMotorControlArray(
            bodyIndex=self.ur_arm.body_id,
            jointIndices=[idx for idx in self.ur_arm.joint_idx_dict.values()],
            controlMode=self.control_mode,
            targetPositions=tgt_pos,
            targetVelocities=self.target_vel,
            forces=self.forces,
            positionGains=self.pos_gains,
            velocityGains=self.vel_gains
        )

