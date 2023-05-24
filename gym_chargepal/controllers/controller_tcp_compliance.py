from __future__ import annotations

# global
import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from rigmopy import utils_math as rp_math
from rigmopy import Vector3d, Vector6d, Pose

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
from gym_chargepal.controllers.controller import Controller, ControllerCfg
from gym_chargepal.utility.spatial_pd_controller import SpatialPDController
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPComplianceControllerCfg(ControllerCfg):
    period: float = -1.0
    gravity: Vector3d = Vector3d().from_xyz([0.0, 0.0, -9.81])
    Kp: npt.NDArray[np.float64] = 500.0 * np.identity(6)
    Kd: npt.NDArray[np.float64] = 350.0 * np.identity(6)


class TCPComplianceController(Controller):

    def __init__(self,
                 config: dict[str, Any],
                 jacobian: Jacobian,
                 ur_arm: URArm,
                 vrt_ur_arm: VirtualURArm,
                 control_interface: JointPositionMotorControl | JointVelocityMotorControl,
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
        # Scale actions
        action[0:3] *= self.cfg.wa_lin
        action[0:6] *= self.cfg.wa_ang
        action_wrt_arm = Vector6d().from_xyzXYZ(action.tolist())

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

        # Get pose of the plug and scale it
        X_goal_wrt_arm = Pose().from_xyz([0.7, 0.5, 0.5]).from_axis_angle([0.0, np.pi/2, 0.0])
        X_plug_wrt_arm = self.plug_sensor.noisy_X_arm2sensor
        q_error_wrt_arm = rp_math.quaternion_difference(X_plug_wrt_arm.q, X_goal_wrt_arm.q)
        p_error_wrt_arm = X_goal_wrt_arm.p - X_plug_wrt_arm.p
        # Get 6d vector
        X_ctrl_wrt_arm = self.cfg.Kp.dot(np.array(p_error_wrt_arm.xyz + q_error_wrt_arm.axis_angle))
        X_ctrl_wrt_arm = Vector6d().from_xyzXYZ(X_ctrl_wrt_arm)

        # # pos_arm2plug = np.array(self.ur_arm.tcp_link.p_world2link.xyz, dtype=np.float64)
        # # eul_arm2plug = np.array(self.ur_arm.tcp_link.q_world2link.to_euler_angle(), dtype=np.float64)
        # pos_arm2plug = self.plug_sensor.noisy_p_arm2sensor.xyz
        # eul_arm2plug = self.plug_sensor.noisy_q_arm2sensor.to_euler_angle()
        
        # pos_arm2goal = np.array([0.8, 0.5, 0.5], dtype=np.float64)
        # eul_arm2goal = np.array([-0.378, np.pi/2, 0.378], dtype=np.float64)

        # pos_error = (pos_arm2goal - pos_arm2plug).tolist()  # type: ignore
        # eul_error = (eul_arm2goal - eul_arm2plug).tolist()  # type: ignore
        # X_error = Vector6d().from_xyzXYZ(self.cfg.Kp.dot(np.array(pos_error + eul_error)))

        # Get the speed of the plug and damp it
        V_goal_wrt_arm = Vector6d()
        V_plug_wrt_arm = self.plug_sensor.noisy_V_wrt_arm
        V_ctrl_wrt_arm = Vector6d().from_xyzXYZ(self.cfg.Kd.dot((V_goal_wrt_arm - V_plug_wrt_arm).to_numpy()))

        # Desired end-effector force
        f_net = action_wrt_arm - ft_wrt_arm + X_ctrl_wrt_arm + V_ctrl_wrt_arm
        # f_net = action_wrt_arm - ft_wrt_arm - V_plug_ctrl_wrt_arm
        f_ctrl = self.spatial_pd_ctrl.update(f_net, self.cfg.period)
        # Get joint configuration
        j_pos = self.ur_arm.joint_pos
        j_vel = self.ur_arm.joint_vel
        j_acc = tuple(6 * [0.0])

        # Get Jacobians
        # jac_t, jac_r = self.jacobian.calculate(j_pos, j_vel, j_acc)
        jac_t, jac_r = self.vrt_ur_arm.jacobian(j_pos, j_vel, j_acc)
        # merge into one jacobian matrix
        # jac_t_wrt_arm = self.ur_arm.q_world2arm.apply(Vector3d().from_xyz(jac_t))
        # jac_r_wrt_arm = self.ur_arm.q_world2arm.apply(Vector3d().from_xyz(jac_r))
        J = np.array(jac_t + jac_r)

        H = self.vrt_ur_arm.calc_inertial_matrix(joint_pos=j_pos)
        fc = f_ctrl.to_numpy()
        # Compute joint accelerations according to \f$ \ddot{q} = H^{-1} ( J^T f) \f$
        joint_acc = inv(H).dot(J.T).dot(fc)
        joint_pos: npt.NDArray[np.float64] = self.joint_pos + self.joint_vel * self.cfg.period
        joint_vel = self.joint_vel + joint_acc * self.cfg.period
        # Limit joint velocities. TODO: Limit velocities in Cartesian space
        clip_value = 0.5
        joint_vel = np.clip(joint_vel, a_min=-clip_value, a_max=clip_value)
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
