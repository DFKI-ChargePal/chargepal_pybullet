from __future__ import annotations

# global
import logging
import numpy as np
from rigmopy import Pose
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.utility.constants import MotionAxis
from gym_chargepal.controllers.controller import Controller, ControllerCfg
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl

# mypy
import numpy.typing as npt
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class TcpPositionControllerCfg(ControllerCfg):
    linear_enabled_motion_axis: tuple[bool, ...] = (True, True, True)
    angular_enabled_motion_axis: tuple[bool, ...] = (True, True, True)
    # Absolute default positions for disabled motion directions
    plug_lin_config: tuple[float, ...] | None = None
    plug_ang_config: tuple[float, ...] | None = None
    # Action scaling
    wa_lin: float = 0.01  # action scaling in linear directions [m]
    wa_ang: float = 0.01 * np.pi  # action scaling in angular directions [rad]


class TcpPositionController(Controller):
    """ Cartesian tool center point position controller """
    def __init__(self,
        config: dict[str, Any],
        arm: URArm,
        ik_solver: IKSolver,
        controller_interface: JointPositionMotorControl
        ) -> None:
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: TcpPositionControllerCfg = TcpPositionControllerCfg()
        self.cfg.update(**config)
        # object references
        self.arm = arm
        self.ik_solver = ik_solver
        self.controller_interface = controller_interface
        # constants
        self.wa_lin = self.cfg.wa_lin
        self.wa_ang = self.cfg.wa_ang
        self.pos0_base2tcp = np.array([0, 0, 0])
        self.ori0_base2tcp = np.array([0, 0, 0])
        # mapping of the enabled motion axis to the indices
        self.pos_motion_axis: dict[bool, list[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.linear_enabled_motion_axis):
            self.pos_motion_axis[mode].append(axis)
        self.ori_motion_axis: dict[bool, list[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.angular_enabled_motion_axis):
            self.ori_motion_axis[mode].append(axis)
        # Slices for the linear and angular actions.
        start_idx = 0
        stop_idx = len(self.pos_motion_axis[MotionAxis.ENABLED])
        self.pos_action_ids = slice(start_idx, stop_idx)
        start_idx = stop_idx
        stop_idx = start_idx + len(self.ori_motion_axis[MotionAxis.ENABLED])
        self.ori_action_ids = slice(start_idx, stop_idx)

    def reset(self) -> None:
        self.pos0_base2tcp = np.array(self.arm.get_p_base2tcp().xyz)
        self.ori0_base2tcp = np.array(self.arm.get_q_base2tcp().xyzw)

    def update(self, action: npt.NDArray[np.float32]) -> None:
        """
        Updates the tcp position controller
        : param action: Action array; The action sequence is defined as (x y z roll pitch yaw).
                        If not all motion directions are enabled, the actions will be executed 
                        in the order in which they are given.
        : return: None
        """
        # Scale action
        action[self.pos_action_ids] *= self.wa_lin
        action[self.ori_action_ids] *= self.wa_ang
        # Get current pose
        pos_base2tcp = np.array(self.arm.get_p_base2tcp().xyz)
        ori_base2tcp = np.array(self.arm.get_q_base2tcp().to_euler_angle())
        # Increment pose by new action
        pos_base2tcp[self.pos_motion_axis[MotionAxis.ENABLED]] += action[self.pos_action_ids]
        ori_base2tcp[self.ori_motion_axis[MotionAxis.ENABLED]] += action[self.ori_action_ids]
        # Set disabled axis to default values to avoid pos drift.
        pos_base2tcp[self.pos_motion_axis[MotionAxis.DISABLED]] = self.pos0_base2tcp[self.pos_motion_axis[MotionAxis.DISABLED]]
        ori_base2tcp[self.ori_motion_axis[MotionAxis.DISABLED]] = self.ori0_base2tcp[self.ori_motion_axis[MotionAxis.DISABLED]]
        # Compose new end-effector pose
        X_base2tcp = Pose().from_xyz(pos_base2tcp).from_euler_angle(ori_base2tcp)
        # Transform to joint space positions
        joint_pos = self.ik_solver.solve(X_base2tcp)
        # Send command to robot
        self.controller_interface.update(joint_pos)
