# global
import copy
import logging

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.config import JOINT_VELOCITY_MOTOR_CONTROL

# mypy
from typing import Dict, Any, Tuple, List


LOGGER = logging.getLogger(__name__)


class JointVelocityMotorControl:
    """ Interface to use PyBullet Velocity Controller. """
    def __init__(self, hyperparams: Dict[str, Any], ur_arm: URArm):
        config: Dict[str, Any] = copy.deepcopy(JOINT_VELOCITY_MOTOR_CONTROL)
        config.update(hyperparams)
        self.hyperparams = config
        # Save references
        self.ur_arm = ur_arm
        # constants
        self.control_mode: int = self.hyperparams['control_mode']
        self.target_pos: List[float] = self.hyperparams['target_pos']
        self.forces: List[float] = self.hyperparams['forces']
        self.pos_gains: List[float] = self.hyperparams['pos_gains']
        self.vel_gains: List[float] = self.hyperparams['vel_gains']

    def update(self, tgt_vel: Tuple[float, ...]) -> None:
        """ Update velocity controllers. Inputs are the desired joint velocities of the arm. """
        self.ur_arm.bc.setJointMotorControlArray(
            bodyIndex=self.ur_arm.body_id,
            jointIndices=[idx for idx in self.ur_arm.joint_idx_dict.values()],
            controlMode=self.control_mode,
            targetPositions=self.target_pos,
            targetVelocities=tgt_vel,
            forces=self.forces,
            positionGains=self.pos_gains,  # No effect in velocity control mode
            velocityGains=self.vel_gains
        )
