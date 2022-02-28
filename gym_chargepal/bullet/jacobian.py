import copy
import logging
import pybullet as p

# mypy
from typing import Dict, Any, Union, Tuple
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.bullet.config import JACOBIAN
from gym_chargepal.bullet.bullet_observer import BulletObserver


LOGGER = logging.getLogger(__name__)


class Jacobian(BulletObserver):
    """ Class to calculate the jacobians of the end effector. """

    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(JACOBIAN)
        config.update(hyperparams)
        self._hyperparams = config
        BulletObserver.__init__(self)
        # get attributes of the world
        self._world = world
        self._world.attach_bullet_obs(self)
        self._physics_client_id = -1
        self._robot_id = -1
        self._ee_tip = -1

    def calculate(self, pos: Tuple[float, ...], vel: Tuple[float, ...], acc: Tuple[float, ...]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Calculate jacobian
        :param pos: joint positions
        :param vel: joint velocities
        :param acc: desired joint accelerations
        :return: translational and rotational jacobians
        """
        # get link state to get the local inertial frame position of the end effector link
        ee_link_state = p.getLinkState(self._robot_id, self._ee_tip, True, True, physicsClientId=self._physics_client_id)
        local_inertial_trn = ee_link_state[2]

        # Important to omit segmentation faults...
        # pos, vel, acc must be of type list or tuple
        jac: Tuple[Tuple[float], Tuple[float]] = p.calculateJacobian(
            bodyUniqueId=self._robot_id,
            linkIndex=self._ee_tip,
            localPosition=local_inertial_trn,
            objPositions=pos,
            objVelocities=vel,
            objAccelerations=acc,
            physicsClientId=self._physics_client_id
        )
        # jac[0] translational jacobian x_dot = J_t * q_dot
        # jac[1] rotational jacobian r_dot = J_r * q_dot
        return jac[0], jac[1]

    def update_bullet_id(self) -> None:
        self._physics_client_id = self._world.physics_client_id
        self._robot_id = self._world.robot_id
        self._ee_tip = self._world.plug_reference_frame_idx
