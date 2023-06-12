from __future__ import annotations
from typing import Any

# global
import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from numpy import typing as npt
from rigmopy import utils_math as rp_math
from rigmopy import Vector3d, Vector6d, Pose, Quaternion


# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
from gym_chargepal.controllers.controller import Controller, ControllerCfg
from gym_chargepal.utility.spatial_pd_controller import SpatialPDController
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl


@dataclass
class TCPMotionControllerCfg(ControllerCfg):
    period: float = -1.0
    error_scale: float = 100.0


class TCPMotionController(Controller):

    def __init__(self, 
                 config: dict[str, Any],
                 jacobian: Jacobian,
                 ur_arm: URArm,
                 vrt_ur_arm: VirtualURArm,
                 control_interface: JointPositionMotorControl | JointVelocityMotorControl,
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point motion controller

        Args:
            config: Dictionary to overwrite configuration values
            jacobian: Jacobian class reference
            ur_arm: URArm class reference
            vrt_ur_arm: VirtualURArm class reference
            control_interface: JointPositionMotorControl or JointVelocityMotorControl class reference
            plug_sensor: PlugSensor class reference
        """
        super().__init__(config=config)
        # Create configuration and overwrite values
        self.cfg: TCPMotionControllerCfg = TCPMotionControllerCfg()
        self.cfg.update(**config)
        config_pd_ctrl = ch.search(config, 'pd_controller')
        self.spatial_pd_ctrl = SpatialPDController(config=config_pd_ctrl)
        # Controller state
        self.X_arm2goal = Pose()
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        # Save references
        self.ur_arm = ur_arm
        self.jacobian = jacobian
        self.vrt_ur_arm = vrt_ur_arm
        self.plug_sensor = plug_sensor
        self.control_interface = control_interface
        if self.cfg.period < 0.0:
            raise ValueError(f"Controller period ({self.cfg.period}) smaller than 0.0."
                             f"Probably not set via config dictionary: {config}")

    def reset(self) -> None:
        if not self.vrt_ur_arm.is_connected:
            self.vrt_ur_arm.connect()
        self.spatial_pd_ctrl.reset()
        self.X_arm2goal = self.plug_sensor.noisy_X_arm2sensor
        self.joint_vel = np.array(self.ur_arm.joint_vel)
        self.joint_pos = np.array(self.ur_arm.joint_pos)

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
        # Get joint configuration
        j_pos = self.ur_arm.joint_pos
        j_vel = self.ur_arm.joint_vel
        j_acc = tuple(6 * [0.0])

        # Get Jacobians
        # jac_t, jac_r = self.jacobian.calculate(j_pos, j_vel, j_acc)
        jac_t, jac_r = self.vrt_ur_arm.jacobian(j_pos, j_vel, j_acc)
        # merge into one jacobian matrix
        J = np.array(jac_t + jac_r)
        H = self.vrt_ur_arm.calc_inertial_matrix(joint_pos=j_pos)
        fc = f_ctrl.to_numpy()
        # Compute joint accelerations according to \f$ \ddot{q} = H^{-1} ( J^T f) \f$
        joint_acc = inv(H).dot(J.T).dot(fc)
        joint_pos: npt.NDArray[np.float64] = self.joint_pos + self.joint_vel * self.cfg.period
        joint_vel = self.joint_vel + joint_acc * self.cfg.period
        # Limit joint velocities. TODO: Limit velocities in Cartesian space
        joint_vel = np.array(self.ur_arm.clip_joint_vel(joint_vel.tolist()))
        # Additional 10 % global damping against unwanted null space motion.
        # This will cause exponential slow-down with action input == 0
        joint_vel *= 0.9

        # Send commands to robot
        if isinstance(self.control_interface, JointVelocityMotorControl):
            self.control_interface.update(tuple(joint_vel))
        elif isinstance(self.control_interface, JointPositionMotorControl):
            self.control_interface.update(tuple(joint_pos))
        else:
            raise TypeError(f"Unknown motor controller of type: {type(self.control_interface)}")
        # Update internal state for next cycle
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
