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
from gym_chargepal.utility.spatial_pd_controller import SpatialPDController
from gym_chargepal.controllers.controller_tcp import TCPController, TCPControllerCfg
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPMotionControllerCfg(TCPControllerCfg):
    error_scale: float = 100.0
    action_scale_lin: float = 0.005
    action_scale_ang: float = 0.005
    spatial_kp: tuple[float, ...] = (2.5, 2.5, 2.5, 2.5, 2.5, 2.5)
    spatial_kd: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


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
        config_pd_ctrl = {
            'kp': self.cfg.spatial_kp,
            'kd': self.cfg.spatial_kd,
        }
        self.spatial_pd_ctrl = SpatialPDController(config=config_pd_ctrl)

    def reset(self) -> None:
        self.spatial_pd_ctrl.reset()
        return super().reset()

    def _compute_motion_error(self, X_plug2goal: Vector6d, X_arm2plug: Pose) -> Vector6d:
        """ Computes the motion error wrt. the robot arm base frame

        Args:
            X_plug2goal: Desired action input between plug and goal (euler angle)
            X_arm2plug: Current pose between arm and plug

        Returns:
            6D motion error wrt. the robot arm base frame
        """
        # Rotate linear action into robot base frame
        q_arm2tcp = self.ur_arm.q_arm2plug
        pos_plug2goal, eul_plug2goal = X_plug2goal.split()
        p_plug2goal = q_arm2tcp.apply(self.cfg.action_scale_lin * pos_plug2goal)
        q_plug2goal = Quaternion().from_euler_angle((self.cfg.action_scale_ang * eul_plug2goal).xyz)

        # Compute spatial error
        p_arm2goal = self.X_arm2goal.p + p_plug2goal
        q_arm2goal = self.X_arm2goal.q * q_plug2goal
        self.X_arm2goal = Pose().from_pq(p_arm2goal, q_arm2goal)
        
        q_error_wrt_arm = rp_math.quaternion_difference(X_arm2plug.q, self.X_arm2goal.q)
        p_error_wrt_arm = self.X_arm2goal.p - X_arm2plug.p

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

        motion_error = Vector6d().from_Vector3d(pos_error, ax_error)
        return motion_error

    def update(self, action: npt.NDArray[np.float32]) -> None:
        """ Concrete update rule of the motion controller

        Args:
            action: Relative motion command
        """
        X_plug2goal = Vector6d().from_xyzXYZ(np.array(action, dtype=np.float64))
        X_arm2plug = self.plug_sensor.noisy_X_arm2sensor
        f_net = self._compute_motion_error(X_plug2goal, X_arm2plug)
        f_ctrl = self.cfg.error_scale * self.spatial_pd_ctrl.update(f_net, self.cfg.period)
        self._to_joint_commands(f_ctrl)
