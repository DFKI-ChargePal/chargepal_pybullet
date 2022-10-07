import logging
import copy
import pybullet as p
import numpy as np

# mypy
from typing import Dict, Any, Union, Tuple, List
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.worlds.world_tdt import WorldTopDownTask

from gym_chargepal.bullet.config import IK_SOLVER
from gym_chargepal.bullet.bullet_observer import BulletObserver
from gym_chargepal.utility.general_utils import wrap

LOGGER = logging.getLogger(__name__)


class IKSolver(BulletObserver):
    """ Inverse kinematics solver. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole, WorldTopDownTask]):
        config: Dict[str, Any] = copy.deepcopy(IK_SOLVER)
        config.update(hyperparams)
        self._hyperparams = config
        BulletObserver.__init__(self)
        # get attributes of the world
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._robot_id = -1
        self._ee_tip = -1
        self._rest_pos: List[float] = []
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
        
        joints: Tuple[float, ...] = p.calculateInverseKinematics(
            bodyUniqueId=self._robot_id,
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
        return joints
        # bring joint configuration in range between - 2 pi and + 2 pi
        # joint_wrapped = tuple([wrap(i, -np.pi, np.pi) for i in joints])
        # return joint_wrapped


    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._robot_id = self._world.robot_id
        self._ee_tip = self._world.plug_reference_frame_idx
        self._rest_pos = [x0 for x0 in self._world.ur_joint_start_config.values()]
