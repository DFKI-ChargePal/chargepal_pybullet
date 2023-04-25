from __future__ import annotations

# global
import logging
import numpy as np
import pybullet as p
from rigmopy import Pose, Vector3d, Quaternion
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.utility.constants import MotionAxis
from gym_chargepal.sensors.sensor_plug import PlugSensor
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
        ik_solver: IKSolver,
        controller_interface: JointPositionMotorControl,
        plug_sensor: PlugSensor
        ) -> None:
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: TcpPositionControllerCfg = TcpPositionControllerCfg()
        self.cfg.update(**config)
        # object references
        self.plug_sensor = plug_sensor
        self.ik_solver = ik_solver
        self.controller_interface = controller_interface
        # constants
        self.wa_lin = self.cfg.wa_lin
        self.wa_ang = self.cfg.wa_ang
        self.plug_lin_config: npt.NDArray[np.float32] | None = None
        self.plug_ang_config: npt.NDArray[np.float32] | None = None
        # mapping of the enabled motion axis to the indices
        self.lin_motion_axis: dict[bool, list[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.linear_enabled_motion_axis):
            self.lin_motion_axis[mode].append(axis)
        self.ang_motion_axis: dict[bool, list[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.angular_enabled_motion_axis):
            self.ang_motion_axis[mode].append(axis)
        # Slices for the linear and angular actions.
        start_idx = 0
        stop_idx = len(self.lin_motion_axis[MotionAxis.ENABLED])
        self.lin_action_ids = slice(start_idx, stop_idx)
        start_idx = stop_idx
        stop_idx = start_idx + len(self.ang_motion_axis[MotionAxis.ENABLED])
        self.ang_action_ids = slice(start_idx, stop_idx)

    def reset(self, X_world2plug: Pose) -> None:
        self.plug_lin_config = np.array(X_world2plug.xyz, np.float32)
        self.plug_ang_config = np.array(X_world2plug.to_euler_angle(), np.float32)

    def update(self, action: npt.NDArray[np.float32]) -> None:
        """
        Updates the tcp position controller
        : param action: Action array; The action sequence is defined as (x y z roll pitch yaw).
                        If not all motion directions are enabled, the actions will be executed 
                        in the order in which they are given.
        : return: None
        """
        # Scale action
        action[self.lin_action_ids] *= self.wa_lin
        action[self.ang_action_ids] *= self.wa_ang
        # Get current pose
        plug_lin_pos = np.array(self.plug_sensor.get_pos().xyz)
        plug_ang_pos = np.array(self.plug_sensor.get_ori().to_euler_angle())
        # Increment pose by action
        plug_lin_pos[self.lin_motion_axis[MotionAxis.ENABLED]] += action[self.lin_action_ids]
        plug_ang_pos[self.ang_motion_axis[MotionAxis.ENABLED]] += action[self.ang_action_ids]
        # Set disabled axis to default values to avoid pos drift.
        reset_error = False
        if self.plug_lin_config is not None:
            plug_lin_pos[self.lin_motion_axis[MotionAxis.DISABLED]] = self.plug_lin_config[
                self.lin_motion_axis[MotionAxis.DISABLED]
                ]
        else:
            reset_error = True
        if self.plug_ang_config is not None:
            plug_ang_pos[self.ang_motion_axis[MotionAxis.DISABLED]] = self.plug_ang_config[
                self.ang_motion_axis[MotionAxis.DISABLED]
                ]
        else:
            reset_error = True
        if reset_error:
            LOGGER.error(f"Please reset controller before use the update function.")

        # Compose new plug pose
        X_arm2plug = Pose().from_xyz(plug_lin_pos).from_euler_angle(plug_ang_pos)
        # Get reference pose
        p_world2arm = self.plug_sensor.ur_arm.tcp.ref_link.get_pos_ref() if self.plug_sensor.ur_arm.tcp.ref_link else Vector3d()
        q_world2arm = self.plug_sensor.ur_arm.tcp.ref_link.get_ori_ref() if self.plug_sensor.ur_arm.tcp.ref_link else Quaternion()
        X_world2arm = Pose().from_pq(p_world2arm, q_world2arm)
        X_world2plug = X_world2arm * X_arm2plug
        # Transform to joint space positions
        joint_pos = self.ik_solver.solve(X_world2plug)
        # Send command to robot
        self.controller_interface.update(joint_pos)
