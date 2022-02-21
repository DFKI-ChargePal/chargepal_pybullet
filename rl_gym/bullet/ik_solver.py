import logging
import copy
import pybullet as p
import numpy as np

# mypy
from typing import Dict, Any, Union, Tuple, List
from chargepal_pybullet.rl_gym.worlds.world_ptp import WorldPoint2Point
from chargepal_pybullet.rl_gym.worlds.world_pih import WorldPegInHole

from chargepal_pybullet.rl_gym.bullet.config import IK_SOLVER
from chargepal_pybullet.rl_gym.utility.general_utils import wrap

LOGGER = logging.getLogger(__name__)


class IKSolver(object):
    """ Inverse kinematics solver. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(IK_SOLVER)
        config.update(hyperparams)
        self._hyperparams = config
        # get attributes of the world
        self._physics_client_id = world.physics_client_id
        self._arm_id = world.arm_id
        self._ee_tip = world.tool_frame_idx
        self._rest_pos = [x0 for x0 in world.joint_x0.values()]
        # constants
        self._lo_limits: List[float] = self._hyperparams['lower_limits']
        self._up_limits: List[float] = self._hyperparams['upper_limits']
        self._j_ranges: List[float] = self._hyperparams['joint_ranges']
        self._max_iter: float = self._hyperparams['max_num_iterations']
        self._residual_threshold: float = self._hyperparams['residual_threshold']

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

        joints = p.calculateInverseKinematics(
            bodyUniqueId=self._arm_id,
            endEffectorLinkIndex=self._ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=self._lo_limits,
            upperLimits=self._up_limits,
            jointRanges=self._j_ranges,
            restPoses=self._rest_pos,
            maxNumIterations=self._max_iter,
            residualThreshold=self._residual_threshold,
            physicsClientId=self._physics_client_id
        )
        # bring joint configuration in range bet
        joint_wrapped = tuple([wrap(i, -np.pi, np.pi) for i in joints])
        return joint_wrapped
