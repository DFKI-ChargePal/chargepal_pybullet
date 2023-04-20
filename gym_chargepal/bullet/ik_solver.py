# global
import logging
from numpy import pi
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
        self.cfg = IKSolverCfg()
        self.cfg.update(**config)

        # get attributes of the world
        self.ur_arm = ur_arm
        self.rest_pos = [x0 for x0 in self.ur_arm.cfg.joint_default_values.values()]

    def solve(self, pose: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> Tuple[float, ...]:
        """
        Calculate joint configuration with inverse kinematics.
        :param pose: spatial pose containing end effector target position and orientation (quaternion)
        :return: joint configuration
        """
        # check for pose shape
        assert len(pose) == 2
        assert len(pose[0]) == 3
        assert len(pose[1]) == 4
        
        joints: Tuple[float, ...] = self.ur_arm.bc.calculateInverseKinematics(
            bodyUniqueId=self.ur_arm.body_id,
            endEffectorLinkIndex=self.ur_arm.tcp.link_idx,
            targetPosition=pose[0],
            targetOrientation=pose[1],
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
