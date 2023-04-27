# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Dict, Any, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class JacobianCfg(ConfigHandler):
    pass


class Jacobian:
    """ Class to calculate the jacobians of the end effector. """
    def __init__(self, config: Dict[str, Any], ur_arm: URArm):
        # Create configuration object and update values
        self.cfg = JacobianCfg()
        self.cfg.update(**config)
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
        # Get the local inertial frame position of the end effector link
        p_ee2inertial = self.ur_arm.tcp_link.get_p_link2inertial()

        # Important to omit segmentation faults...
        # pos, vel, acc must be of type list or tuple
        jac: Tuple[Tuple[float], Tuple[float]] = self.ur_arm.bullet_client.calculateJacobian(
            bodyUniqueId=self.ur_arm.bullet_body_id,
            linkIndex=self.ur_arm.tcp_link.idx,
            localPosition=p_ee2inertial.xyz,
            objPositions=pos,
            objVelocities=vel,
            objAccelerations=acc,
        )
        # jac[0] translational jacobian x_dot = J_t * q_dot
        # jac[1] rotational jacobian r_dot = J_r * q_dot
        return jac[0], jac[1]
