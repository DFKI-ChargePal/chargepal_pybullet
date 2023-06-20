from __future__ import annotations

# global
import abc
import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass
from rigmopy import Pose, Vector3d, Vector6d

# local
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.bullet.ur_arm import URArm
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
class TCPControllerCfg(ControllerCfg):
    period: float = -1.0
    gravity: Vector3d = Vector3d().from_xyz([0.0, 0.0, -9.81])


class TCPController(Controller):

    def __init__(self, 
                 config: dict[str, Any],
                 ur_arm: URArm,
                 vrt_ur_arm: VirtualURArm,
                 control_interface: JointPositionMotorControl | JointVelocityMotorControl,
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point controller

        Args:
            config: Dictionary to overwrite configuration values
            ur_arm: URArm class reference
            vrt_ur_arm: VirtualURArm class reference
            control_interface: JointPositionMotorControl or JointVelocityMotorControl class reference
            plug_sensor: PlugSensor class reference
        """
        super().__init__(config=config)
                # Create configuration and overwrite values
        self.cfg: TCPControllerCfg = TCPControllerCfg()
        self.cfg.update(**config)
        config_pd_ctrl = ch.search(config, 'pd_controller')
        self.spatial_pd_ctrl = SpatialPDController(config=config_pd_ctrl)
        # Controller state
        self.X_arm2goal = Pose()
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        # Save references
        self.ur_arm = ur_arm
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

    @abc.abstractmethod
    def update(self, action: npt.NDArray[np.float32]) -> None:
        raise NotImplementedError('Must be implemented in subclass.')
    
    def _to_joint_commands(self, f_ctrl: Vector6d) -> None:
        """ Map control inputs in joint space and integrate it and apply it to the robot.

        Args:
            f_ctrl: Force control inputs
        """
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
