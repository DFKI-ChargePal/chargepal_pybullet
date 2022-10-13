import logging
import copy

# mypy
from typing import Dict, Any, Union, Tuple, List
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.worlds.world_tdt import WorldTopDownTask

from gym_chargepal.bullet.config import IK_SOLVER
from gym_chargepal.utility.general_utils import wrap

LOGGER = logging.getLogger(__name__)


class IKSolver:
    """ Inverse kinematics solver. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole, WorldTopDownTask]):
        config: Dict[str, Any] = copy.deepcopy(IK_SOLVER)
        config.update(hyperparams)
        self._hyperparams = config
        # get attributes of the world
        self.world = world
        self.rest_pos = [x0 for x0 in self.world.ur_joint_start_config.values()]
        # constants
        self.lo_limits: List[float] = self._hyperparams['lower_limits']
        self.up_limits: List[float] = self._hyperparams['upper_limits']
        self.j_ranges: List[float] = self._hyperparams['joint_ranges']
        self.max_iter: float = self._hyperparams['max_num_iterations']
        self.residual_threshold: float = self._hyperparams['residual_threshold']

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
        
        joints: Tuple[float, ...] = self.world.bullet_client.calculateInverseKinematics(
            bodyUniqueId=self.world.robot_id,
            endEffectorLinkIndex=self.world.plug_reference_frame_idx,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=self.lo_limits,
            upperLimits=self.up_limits,
            jointRanges=self.j_ranges,
            restPoses=self.rest_pos,
            maxNumIterations=self.max_iter,
            residualThreshold=self.residual_threshold
        )
        return joints
        # bring joint configuration in range between - 2 pi and + 2 pi
        # joint_wrapped = tuple([wrap(i, -np.pi, np.pi) for i in joints])
        # return joint_wrapped
