import copy
import logging

# mypy
from typing import Dict, Any, Tuple
from gym_chargepal.bullet.ur_arm import URArm

from gym_chargepal.bullet.config import JACOBIAN


LOGGER = logging.getLogger(__name__)


class Jacobian:
    """ Class to calculate the jacobians of the end effector. """

    def __init__(self, hyperparams: Dict[str, Any], ur_arm: URArm):
        config: Dict[str, Any] = copy.deepcopy(JACOBIAN)
        config.update(hyperparams)
        self.hyperparams = config
        # Save references
        self.ur_arm = ur_arm

    def calculate(self, pos: Tuple[float, ...], vel: Tuple[float, ...], acc: Tuple[float, ...]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Calculate jacobian
        :param pos: joint positions
        :param vel: joint velocities
        :param acc: desired joint accelerations
        :return: translational and rotational jacobians
        """
        # get link state to get the local inertial frame position of the end effector link
        ee_link_state = self.ur_arm.bc.getLinkState(self.ur_arm.body_id, self.ur_arm.tcp.link_idx, True, True)
        local_inertial_trn = ee_link_state[2]

        # Important to omit segmentation faults...
        # pos, vel, acc must be of type list or tuple
        jac: Tuple[Tuple[float], Tuple[float]] = self.ur_arm.bc.calculateJacobian(
            bodyUniqueId=self.ur_arm.body_id,
            linkIndex=self.ur_arm.tcp.link_idx,
            localPosition=local_inertial_trn,
            objPositions=pos,
            objVelocities=vel,
            objAccelerations=acc,
        )
        # jac[0] translational jacobian x_dot = J_t * q_dot
        # jac[1] rotational jacobian r_dot = J_r * q_dot
        return jac[0], jac[1]
