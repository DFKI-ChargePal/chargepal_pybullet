from __future__ import annotations

# global
import numpy as np
from rigmopy import Vector6d
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
from gym_chargepal.utility.spatial_pd_controller import SpatialPDController
from gym_chargepal.controllers.controller_tcp_force import TCPForceController
from gym_chargepal.controllers.controller_tcp_motion import TCPMotionController
from gym_chargepal.controllers.controller_tcp import TCPController, TCPControllerCfg
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

# typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPComplianceControllerCfg(TCPControllerCfg):
    error_scale: float = 1.0
    force_action_scale_lin: float = 1.0
    force_action_scale_ang: float = 3.0
    motion_action_scale_lin: float = 0.002
    motion_action_scale_ang: float = 0.005
    spatial_kp: tuple[float, ...] = (1e0, 1e0, 1e0, 1e0, 1e0, 1e0)
    spatial_kd: tuple[float, ...] = (1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)
    ee_stiffness: tuple[float, ...] = (3e2, 3e2, 3e2, 3e2, 3e2, 3e2)


class TCPComplianceController(TCPController):

    def __init__(self, 
                 config: dict[str, Any], 
                 ur_arm: URArm, vrt_ur_arm: 
                 VirtualURArm, control_interface: JointPositionMotorControl | JointVelocityMotorControl, 
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point compliance controller. A combination of the force and the motion controller

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
        config_force_ctrl = dict(config)
        config_force_ctrl['action_scale_lin'] = self.cfg.force_action_scale_lin
        config_force_ctrl['action_scale_ang'] = self.cfg.force_action_scale_ang
        self.force_ctrl = TCPForceController(config_force_ctrl, ur_arm, vrt_ur_arm, control_interface, plug_sensor)
        config_motion_ctrl = dict(config)
        config_motion_ctrl['action_scale_lin'] = self.cfg.motion_action_scale_lin
        config_motion_ctrl['action_scale_ang'] = self.cfg.motion_action_scale_ang
        self.motion_ctrl = TCPMotionController(config_motion_ctrl, ur_arm, vrt_ur_arm, control_interface, plug_sensor)
        self.motion_stiffness = np.diag(self.cfg.ee_stiffness)

    def reset(self) -> None:
        self.force_ctrl.reset()
        self.motion_ctrl.reset()
        self.spatial_pd_ctrl.reset()
        return super().reset()
    
    def update(self, action: npt.NDArray[np.float32]) -> None:
        """ Concrete update rule of the compliance controller

        Args:
            action: Combination of motion and force command
        """
        # Get motion error
        X_plug2goal = Vector6d().from_xyzXYZ(np.array(action[0:6], dtype=np.float64))
        X_arm2plug = self.plug_sensor.noisy_X_arm2sensor
        motion_error = Vector6d().from_xyzXYZ(
            self.motion_stiffness.dot(self.motion_ctrl.compute_motion_error(X_plug2goal, X_arm2plug).to_numpy()))
        # Get force error
        fta_plug = Vector6d().from_xyzXYZ(np.array(action[6:12], dtype=np.float64))
        fts_sensor = self.ur_arm.wrench
        force_error = self.force_ctrl.compute_ft_error(fta_plug, fts_sensor)
        f_net = motion_error + force_error
        f_ctrl = self.cfg.error_scale * self.spatial_pd_ctrl.update(f_net, self.cfg.period)
        self._to_joint_commands(f_ctrl)
