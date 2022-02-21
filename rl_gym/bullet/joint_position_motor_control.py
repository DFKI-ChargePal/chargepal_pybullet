import copy
import logging
import pybullet as p

# mypy
from typing import Dict, Any, Union, Tuple, List
from chargepal_pybullet.rl_gym.worlds.world_ptp import WorldPoint2Point
from chargepal_pybullet.rl_gym.worlds.world_pih import WorldPegInHole


from chargepal_pybullet.rl_gym.bullet.config import JOINT_POSITION_MOTOR_CONTROL


LOGGER = logging.getLogger(__name__)


class JointPositionMotorControl(object):
    """ Interface to use PyBullet Position Controller. """
    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(JOINT_POSITION_MOTOR_CONTROL)
        config.update(hyperparams)
        self._hyperparams = config
        # get attributes of the world
        self._physics_client_id = world.physics_client_id
        self._arm_id = world.arm_id
        self._joint_ids = [idx for idx in world.joint_idx.values()]
        # constants
        self._control_mode: int = self._hyperparams['control_mode']
        self._target_vel: List[float] = self._hyperparams['target_vel']
        self._forces: List[float] = self._hyperparams['forces']
        self._pos_gains: List[float] = self._hyperparams['pos_gains']
        self._vel_gains: List[float] = self._hyperparams['vel_gains']

    def update(self, tgt_pos: Tuple[float, ...]) -> None:
        """ Update position controllers. Inputs are the desired joint angels of the arm. """
        assert len(self._joint_ids) == len(tgt_pos)

        p.setJointMotorControlArray(
            bodyIndex=self._arm_id,
            jointIndices=self._joint_ids,
            controlMode=self._control_mode,
            targetPositions=tgt_pos,
            targetVelocities=self._target_vel,
            forces=self._forces,
            positionGains=self._pos_gains,
            velocityGains=self._vel_gains,
            physicsClientId=self._physics_client_id
        )
