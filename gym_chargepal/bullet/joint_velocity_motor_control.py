import copy
import logging
import pybullet as p

# mypy
from typing import Dict, Any, Union, Tuple, List
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.bullet.config import JOINT_VELOCITY_MOTOR_CONTROL
from gym_chargepal.bullet.bullet_observer import BulletObserver


LOGGER = logging.getLogger(__name__)


class JointVelocityMotorControl(BulletObserver):
    """ Interface to use PyBullet Velocity Controller. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(JOINT_VELOCITY_MOTOR_CONTROL)
        config.update(hyperparams)
        self._hyperparams = config
        BulletObserver.__init__(self)
        # get attributes of the world
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._robot_id = -1
        self._joint_ids: List[int] = []
        # constants
        self._control_mode: int = self._hyperparams['control_mode']
        self._target_pos: List[float] = self._hyperparams['target_pos']
        self._forces: List[float] = self._hyperparams['forces']
        self._pos_gains: List[float] = self._hyperparams['pos_gains']
        self._vel_gains: List[float] = self._hyperparams['vel_gains']

    def update(self, tgt_vel: Tuple[float, ...]) -> None:
        """ Update velocity controllers. Inputs are the desired joint velocities of the arm. """
        assert len(self._joint_ids) == len(tgt_vel)

        p.setJointMotorControlArray(
            bodyIndex=self._robot_id,
            jointIndices=self._joint_ids,
            controlMode=self._control_mode,
            targetPositions=self._target_pos,
            targetVelocities=tgt_vel,
            forces=self._forces,
            positionGains=self._pos_gains,  # No effect in velocity control mode
            velocityGains=self._vel_gains,
            physicsClientId=self._physics_client_id
        )

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._robot_id = self._world.robot_id
        self._joint_ids = [idx for idx in self._world.ur_joint_idx_dict.values()]
