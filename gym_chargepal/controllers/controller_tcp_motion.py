from __future__ import annotations

# global
import numpy as np
from dataclasses import dataclass
from rigmopy import utils_math as rp_math
from rigmopy import Vector3d, Vector6d, Pose, Quaternion


# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
from gym_chargepal.controllers.controller_tcp import TCPController, TCPControllerCfg
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPMotionControllerCfg(TCPControllerCfg):
    error_scale: float = 100.0


class TCPMotionController(TCPController):

    def __init__(self, 
                 config: dict[str, Any],
                 ur_arm: URArm,
                 vrt_ur_arm: VirtualURArm,
                 control_interface: JointPositionMotorControl | JointVelocityMotorControl,
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point motion controller

        Args:
            config: Dictionary to overwrite configuration values
            ur_arm: URArm class reference
            vrt_ur_arm: VirtualURArm class reference
            control_interface: JointPositionMotorControl or JointVelocityMotorControl class reference
            plug_sensor: PlugSensor class reference
        """
        super().__init__(config, ur_arm, vrt_ur_arm, control_interface, plug_sensor)
        # Create configuration and overwrite values
        self.cfg: TCPMotionControllerCfg = TCPMotionControllerCfg()
        self.cfg.update(**config)

    def update(self, action: npt.NDArray[np.float32]) -> None:
        # Compute motion error wrt robot arm base frame

        # Rotate action into robot base frame
        q_arm2tcp = self.ur_arm.q_arm2plug
        p_action = q_arm2tcp.apply(Vector3d().from_xyz(action[0:3] * self.cfg.wa_lin))
        q_action = Quaternion().from_euler_angle(action[3:6] * self.cfg.wa_ang)

        # Compute spatial error
        p_arm2goal = self.X_arm2goal.p + p_action
        q_arm2goal = self.X_arm2goal.q * q_action
        self.X_arm2goal = Pose().from_pq(p_arm2goal, q_arm2goal)
        X_arm2tcp = self.plug_sensor.noisy_X_arm2sensor
        q_error_wrt_arm = rp_math.quaternion_difference(X_arm2tcp.q, self.X_arm2goal.q)
        p_error_wrt_arm = self.X_arm2goal.p - X_arm2tcp.p

        r_error = np.array(q_error_wrt_arm.axis_angle)

        # Angles error always within [0,Pi)
        angle_error = np.max(np.abs(r_error))
        if angle_error < 1e7:
            axis_error = r_error
        else:
            axis_error = r_error/angle_error
        # Clamp maximal tolerated error.
        # The remaining error will be handled in the next control cycle.
        # Note that this is also the maximal offset that the
        # cartesian_compliance_controller can use to build up a restoring stiffness
        # wrench.
        angle_error = np.clip(angle_error, 0.0, 1.0)
        ax_error = Vector3d().from_xyz(angle_error * axis_error)
        distance_error = np.clip(p_error_wrt_arm.magnitude, -1.0, 1.0)
        pos_error = distance_error * p_error_wrt_arm

        error = self.cfg.error_scale * Vector6d().from_Vector3d(pos_error, ax_error)
        f_net = error
        f_ctrl = self.spatial_pd_ctrl.update(f_net, self.cfg.period)
        self._to_joint_commands(f_ctrl)
