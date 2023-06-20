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
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPComplianceControllerCfg(TCPControllerCfg):
    gravity: Vector3d = Vector3d().from_xyz([0.0, 0.0, -9.81])
    wa_force: float = 1e2
    wa_torque: float = np.pi * 1e2
    spatial_kp: tuple[float, ...] = (0.09, 0.09, 0.09, 0.9, 0.9, 0.9)
    spatial_kd: tuple[float, ...] = (1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    Kp_lin: npt.NDArray[np.float64] = 1.85e1 * np.identity(3)  # max 1e5 * np.identity(3)
    Kp_ang: npt.NDArray[np.float64] = 1.25e2 * np.identity(3)  # max 1e3 * np.identity(3)
    Kd: npt.NDArray[np.float64] = 0.1 * np.identity(6)


class TCPComplianceController(TCPController):

    def __init__(self,
                 config: dict[str, Any],
                 ur_arm: URArm,
                 vrt_ur_arm: VirtualURArm,
                 control_interface: JointPositionMotorControl | JointVelocityMotorControl,
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point compliance controller

        Args:
            config: Dictionary to overwrite configuration values
            ur_arm: URArm class reference
            vrt_ur_arm: VirtualURArm class reference
            control_interface: JointPositionMotorControl or JointVelocityMotorControl class reference
            plug_sensor: PlugSensor class reference
        """
        super().__init__(config, ur_arm, vrt_ur_arm, control_interface, plug_sensor)
        # Create configuration and overwrite values
        self.cfg: TCPComplianceControllerCfg = TCPComplianceControllerCfg()
        self.cfg.update(**config)
        config_pd_ctrl = {
            'kp': self.cfg.spatial_kp,
            'kd': self.cfg.spatial_kd,
        }
        self.spatial_pd_ctrl = SpatialPDController(config=config_pd_ctrl)

    def reset(self) -> None:
        self.spatial_pd_ctrl.reset()
        return super().reset()

    def update(self, action: npt.NDArray[np.float32]) -> None:
        # Split action and scale it
        # Wrench action
        # action[6:9] *= self.cfg.wa_force
        # action[9:12] *= self.cfg.wa_torque
        F_action = Vector6d()  # .from_xyzXYZ(action[6:12])
        
        # Compute motion error wrt robot arm base frame
        q_arm2tcp = self.ur_arm.q_arm2plug
        # Rotate action into robot base frame
        p_action = q_arm2tcp.apply(Vector3d().from_xyz(action[0:3] * self.cfg.wa_lin))
        q_action = Quaternion().from_euler_angle(action[3:6] * self.cfg.wa_ang)

        # Spatial action
        # p_action = Vector3d().from_xyz(action[0:3] * self.cfg.wa_lin)
        # q_action = Quaternion().from_euler_angle(action[3:6] * self.cfg.wa_ang)

        p_arm2goal = self.X_arm2goal.p + p_action
        q_arm2goal = self.X_arm2goal.q * q_action
        self.X_arm2goal = Pose().from_pq(p_arm2goal, q_arm2goal)
        X_arm2tcp = self.plug_sensor.noisy_X_arm2sensor
        q_error_wrt_arm = rp_math.quaternion_difference(X_arm2tcp.q, self.X_arm2goal.q)
        p_error_wrt_arm = (self.X_arm2goal.p - X_arm2tcp.p).normalize()
        # X_goal_wrt_arm = Pose().from_pq(p_arm2goal, q_arm2goal)
        # self.X_arm2goal = self.plug_sensor.noisy_X_arm2sensor
        # q_error_wrt_arm = rp_math.quaternion_difference(self.X_arm2goal.q, X_goal_wrt_arm.q)
        # p_error_wrt_arm = X_goal_wrt_arm.p - self.X_arm2goal.p
        # Get 6d vector
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
        angle_error = np.clip(angle_error, -1.0, 1.0)
        axis_error = angle_error * axis_error

        r_ctrl_wrt_arm = Vector3d().from_xyz(self.cfg.Kp_ang.dot(axis_error))
        x_ctrl_wrt_arm = Vector3d().from_xyz(self.cfg.Kp_lin.dot(np.array(p_error_wrt_arm.xyz)))
        X_ctrl_wrt_arm = Vector6d().from_Vector3d(x_ctrl_wrt_arm, r_ctrl_wrt_arm)

        # Get latest force readings
        ft_wrt_ft = self.ur_arm.raw_wrench.split()
        q_ft2arm = self.ur_arm.q_arm2fts.inverse()

        # Rotate force torque measurements into robot base
        f_wrt_arm = q_ft2arm.apply(ft_wrt_ft[0])
        t_wrt_arm = q_ft2arm.apply(ft_wrt_ft[1])

        # Compensate the force of the sensor gravity
        sensor_mass = self.ur_arm.fts_mass
        f_sensor_weight_wrt_world = sensor_mass * self.cfg.gravity
        f_sensor_weight_wrt_arm = self.ur_arm.q_world2arm.apply(f_sensor_weight_wrt_world)
        p_sensor2com_wrt_arm = self.ur_arm.fts_com_wrt_sensor
        t_sensor_weight_wrt_arm = Vector3d().from_xyz(np.cross(f_sensor_weight_wrt_arm.xyz, p_sensor2com_wrt_arm.xyz))

        # Compensate the force of the plug gravity
        plug_mass = self.ur_arm.tool_mass
        f_plug_weight_wrt_world = plug_mass * self.cfg.gravity
        f_plug_weight_wrt_arm = self.ur_arm.q_world2arm.apply(f_plug_weight_wrt_world)
        # p_arm2sensor_wrt_arm = self.ur_arm.p_arm2fts
        # p_sensor2plug_com_wrt_arm = self.ur_arm.tool_com_wrt_arm
        # p_sensor2plug_com_wrt_arm = p_arm2plug_com_wrt_arm - p_arm2sensor_wrt_arm
        t_plug_weight_wrt_arm = Vector3d().from_xyz(np.cross(f_plug_weight_wrt_arm.xyz, p_sensor2com_wrt_arm.xyz))

        # Add compensation to measurements
        f_wrt_arm -= f_sensor_weight_wrt_arm + f_plug_weight_wrt_arm
        t_wrt_arm -= t_sensor_weight_wrt_arm + t_plug_weight_wrt_arm

        # Merge force and torque signal
        ft_wrt_arm = Vector6d().from_Vector3d(f_wrt_arm, t_wrt_arm)

        # Get the speed of the plug and damp it
        V_goal_wrt_arm = Vector6d()
        V_plug_wrt_arm = self.plug_sensor.noisy_V_wrt_arm
        V_ctrl_wrt_arm = Vector6d().from_xyzXYZ(self.cfg.Kd.dot((V_goal_wrt_arm - V_plug_wrt_arm).to_numpy()))

        # Desired end-effector force
        f_net = X_ctrl_wrt_arm # F_action - ft_wrt_arm + X_ctrl_wrt_arm ## + V_ctrl_wrt_arm
        # f_net = action_wrt_arm - ft_wrt_arm - V_plug_ctrl_wrt_arm
        f_ctrl = self.spatial_pd_ctrl.update(f_net, self.cfg.period)
        self._to_joint_commands(f_ctrl)
