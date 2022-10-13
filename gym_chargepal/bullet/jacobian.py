import copy
import logging

# mypy
from typing import Dict, Any, Union, Tuple
from gym_chargepal.worlds.world_ptp import WorldPoint2Point
from gym_chargepal.worlds.world_pih import WorldPegInHole

from gym_chargepal.bullet.config import JACOBIAN


LOGGER = logging.getLogger(__name__)


class Jacobian:
    """ Class to calculate the jacobians of the end effector. """

    def __init__(self, hyperparams: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        config: Dict[str, Any] = copy.deepcopy(JACOBIAN)
        config.update(hyperparams)
        self.hyperparams = config
        # get attributes of the world
        self.world = world
        # self._physics_client_id = -1
        # self._robot_id = -1
        # self._ee_tip = -1

    def calculate(self, pos: Tuple[float, ...], vel: Tuple[float, ...], acc: Tuple[float, ...]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Calculate jacobian
        :param pos: joint positions
        :param vel: joint velocities
        :param acc: desired joint accelerations
        :return: translational and rotational jacobians
        """
        # get link state to get the local inertial frame position of the end effector link
        ee_link_state = self.world.bullet_client.getLinkState(self.world.robot_id, self.world.plug_reference_frame_idx, True, True)
        local_inertial_trn = ee_link_state[2]

        # Important to omit segmentation faults...
        # pos, vel, acc must be of type list or tuple
        jac: Tuple[Tuple[float], Tuple[float]] = self.world.bullet_client.calculateJacobian(
            bodyUniqueId=self.world.robot_id,
            linkIndex=self.world.plug_reference_frame_idx,
            localPosition=local_inertial_trn,
            objPositions=pos,
            objVelocities=vel,
            objAccelerations=acc,
        )
        # jac[0] translational jacobian x_dot = J_t * q_dot
        # jac[1] rotational jacobian r_dot = J_r * q_dot
        return jac[0], jac[1]
