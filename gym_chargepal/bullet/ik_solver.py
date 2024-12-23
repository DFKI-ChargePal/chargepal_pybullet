# global
import logging
from numpy import pi
from rigmopy import Pose
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.utility.general_utils import wrap
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Any, Dict, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class IKSolverCfg(ConfigHandler):
    lower_limits: Tuple[float, ...] = (-2*pi, -2*pi, -2*pi, -2*pi, -2*pi, -2*pi)
    upper_limits: Tuple[float, ...] = (2*pi, 2*pi, 2*pi, 2*pi, 2*pi, 2*pi)
    joint_ranges: Tuple[float, ...] = (2*pi, 2*pi, 2*pi, 2*pi, 2*pi, 2*pi)
    max_num_iterations: int = 100
    residual_threshold: float = 1e-7


class IKSolver:
    """ Inverse kinematics solver. """
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Create configuration and override values
        self.cfg = IKSolverCfg()
        self.cfg.update(**config)

        # Get attributes of the world
        self.ur_arm = ur_arm
        self.rest_pos = [x0 for x0 in self.ur_arm.cfg.joint_default_values.values()]

    def solve(self, pose: Pose) -> Tuple[float, ...]:
        """
        Calculate joint configuration with inverse kinematics.
        :param pose: spatial pose containing end effector target position and orientation (quaternion)
        :return: joint configuration
        """
        joints: Tuple[float, ...] = self.ur_arm.bullet_client.calculateInverseKinematics(
            bodyUniqueId=self.ur_arm.bullet_body_id,
            endEffectorLinkIndex=self.ur_arm.tcp_link.idx,
            targetPosition=pose.xyz,
            targetOrientation=pose.xyzw,
            lowerLimits=self.cfg.lower_limits,
            upperLimits=self.cfg.upper_limits,
            jointRanges=self.cfg.joint_ranges,
            restPoses=self.rest_pos,
            maxNumIterations=self.cfg.max_num_iterations,
            residualThreshold=self.cfg.residual_threshold
        )
        return joints
        # bring joint configuration in range between - 2 pi and + 2 pi
        # joint_wrapped = tuple([wrap(i, -np.pi, np.pi) for i in joints])
        # return joint_wrapped
