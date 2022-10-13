import copy
import logging

# mypy
from typing import Dict, Any, Union, Tuple, List
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.worlds.world_tdt import WorldTopDownTask

from gym_chargepal.bullet.config import JOINT_POSITION_MOTOR_CONTROL


LOGGER = logging.getLogger(__name__)


class JointPositionMotorControl:
    """ Interface to use PyBullet Position Controller. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole, WorldTopDownTask]):
        config: Dict[str, Any] = copy.deepcopy(JOINT_POSITION_MOTOR_CONTROL)
        config.update(hyperparams)
        self.hyperparams = config
        # get attributes of the world
        self.world = world
        # constants
        self.control_mode: int = self.hyperparams['control_mode']
        self.target_vel: List[float] = self.hyperparams['target_vel']
        self.forces: List[float] = self.hyperparams['forces']
        self.pos_gains: List[float] = self.hyperparams['pos_gains']
        self.vel_gains: List[float] = self.hyperparams['vel_gains']

    def update(self, tgt_pos: Tuple[float, ...]) -> None:
        """ Update position controllers. Inputs are the desired joint angels of the arm. """
        self.world.bullet_client.setJointMotorControlArray(
            bodyIndex=self.world.robot_id,
            jointIndices=[idx for idx in self.world.ur_joint_idx_dict.values()],
            controlMode=self.control_mode,
            targetPositions=tgt_pos,
            targetVelocities=self.target_vel,
            forces=self.forces,
            positionGains=self.pos_gains,
            velocityGains=self.vel_gains
        )

