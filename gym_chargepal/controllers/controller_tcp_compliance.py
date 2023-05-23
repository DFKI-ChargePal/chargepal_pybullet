from __future__ import annotations

# global
import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from rigmopy import Vector3d, Vector6d

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.sensors.sensor_ft import FTSensor
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
from gym_chargepal.controllers.controller import Controller, ControllerCfg
from gym_chargepal.utility.spatial_pd_controller import SpatialPDController
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPComplianceControllerCfg(ControllerCfg):
    gravity: Vector3d = Vector3d().from_xyz([0.0, 0.0, -9.81])
    p_gain: npt.NDArray[np.float64] = 10000.0 * np.identity(6)
    v_damping: npt.NDArray[np.float64] = 0.1 * np.identity(3, dtype=np.float64)


class TCPComplianceController(Controller):

    def __init__(self,
                 config: dict[str, Any],
                 jacobian: Jacobian,
                 ur_arm: URArm,
                 vrt_ur_arm: VirtualURArm,
                 control_interface: JointVelocityMotorControl,
                 ft_sensor: FTSensor,
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point compliance controller

        Args:
            config: Dictionary to overwrite configuration values
        """
        super().__init__(config=config)
        # Create configuration and overwrite values
        self.cfg: TCPComplianceControllerCfg = TCPComplianceControllerCfg()
        self.cfg.update(**config)
        self.spatial_pd_ctrl = SpatialPDController(config=config)
        self.period = self.spatial_pd_ctrl.cfg.period
        # Save references
        self.ur_arm = ur_arm
        self.jacobian = jacobian
        self.ft_sensor = ft_sensor
        self.vtr_ur_arm = vrt_ur_arm
        self.plug_sensor = plug_sensor
        self.control_interface = control_interface

    def reset(self) -> None:
        if not self.vtr_ur_arm.is_connected:
            self.vtr_ur_arm.connect()
        self.spatial_pd_ctrl.reset()
        self.joint_vel = np.zeros(6)
        self.joint_pos = np.zeros(6)

    def update(self, action: npt.NDArray[np.float32]) -> None:
        # Scale actions
        action[0:3] *= self.cfg.wa_lin
        action[0:6] *= self.cfg.wa_ang
        action_wrt_arm = Vector6d().from_xyzXYZ(action.tolist())

        # Get latest force readings
        noisy_ft_wrt_ft = self.ft_sensor.noisy_wrench.split()
        noisy_q_ft2arm = self.ft_sensor.noisy_q_arm2sensor.inverse()

        # Rotate force torque measurements into robot base
        noisy_f_wrt_arm = noisy_q_ft2arm.apply(noisy_ft_wrt_ft[0])
        noisy_t_wrt_arm = noisy_q_ft2arm.apply(noisy_ft_wrt_ft[1])

        # Compensate the force of the sensor gravity
        sensor_mass = self.ur_arm.fts_mass
        f_sensor_weight_wrt_world = sensor_mass * self.cfg.gravity
        f_sensor_weight_wrt_arm = self.ur_arm.q_world2arm.apply(f_sensor_weight_wrt_world)
        p_sensor2com_wrt_arm = self.ur_arm.fts_com
        t_sensor_weight_wrt_arm = np.cross(p_sensor2com_wrt_arm.xyz, f_sensor_weight_wrt_arm.xyz)

        # Compensate the force of the plug gravity
        plug_mass = self.ur_arm.tool_mass
        f_plug_weight_wrt_world = plug_mass * self.cfg.gravity
        f_plug_weight_wrt_arm = self.ur_arm.q_world2arm.apply(f_plug_weight_wrt_world)
        p_plug2com_wrt_arm = self.ur_arm.tool_com
        t_plug_weight_wrt_arm = np.cross(p_plug2com_wrt_arm.xyz, f_plug_weight_wrt_arm.xyz)

        # Add compensation to measurements
        noisy_f_wrt_arm += f_sensor_weight_wrt_arm + f_plug_weight_wrt_arm
        noisy_t_wrt_arm += t_sensor_weight_wrt_arm + t_plug_weight_wrt_arm

        # Merge force and torque signal
        noisy_ft_wrt_arm = Vector6d().from_Vector3d(noisy_f_wrt_arm, noisy_t_wrt_arm)

        # Get pose of the plug and scale it
        pos_arm2plug = self.plug_sensor.noisy_p_arm2sensor.xyz
        eul_arm2plug = self.plug_sensor.noisy_q_arm2sensor.to_euler_angle()
        pose_arm2plug_ctrl = Vector6d().from_xyzXYZ(
            self.cfg.p_gain.dot(np.array(pos_arm2plug + eul_arm2plug, dtype=np.float64)))

        # Get the speed of the plug and damp it
        V_plug_ctrl_wrt_arm = Vector6d().from_xyzXYZ(
            self.cfg.v_damping.dot(self.plug_sensor.noisy_V_wrt_arm.to_numpy()))


        # Desired end-effector force
        f_net = action_wrt_arm - noisy_ft_wrt_arm + pose_arm2plug_ctrl - V_plug_ctrl_wrt_arm
        f_ctrl = self.spatial_pd_ctrl.update(f_net)

        # Get joint configuration
        j_pos = self.ur_arm.joint_pos
        j_vel = self.ur_arm.joint_vel
        j_acc = tuple(6 * [0.0])

        # Get Jacobians
        jac_t, jac_r = self.jacobian.calculate(j_pos, j_vel, j_acc)
        # merge into one jacobian matrix
        J = np.array(jac_t + jac_r)

        H = self.vtr_ur_arm.calc_inertial_matrix(joint_pos=j_pos)
        fc = f_ctrl.to_numpy()
        # Compute joint accelerations according to \f$ \ddot{q} = H^{-1} ( J^T f) \f$
        joint_acc = inv(H).dot(J.T).dot(fc)
        joint_pos: npt.NDArray[np.float64] = self.joint_pos + self.joint_vel * self.period
        joint_vel = self.joint_vel + joint_acc * self.period
        # Limit joint velocities. TODO: Limit velocities in Cartesian space
        clip_value = 0.25
        joint_vel = np.clip(joint_vel, a_min=-clip_value, a_max=clip_value)
        # Additional 10 % global damping against unwanted null space motion.
        # This will cause exponential slow-down with action input == 0
        joint_vel *= 0.9

        # Send commands to robot
        self.control_interface.update(tuple(joint_vel))

        # Update internal state for next cycle
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
