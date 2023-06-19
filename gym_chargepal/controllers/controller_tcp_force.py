from __future__ import annotations

# global
import numpy as np
from dataclasses import dataclass
from rigmopy import Vector3d

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ur_arm_virtual import VirtualURArm
from gym_chargepal.controllers.controller_tcp import TCPController, TCPControllerCfg
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

#typing
from typing import Any
from numpy import typing as npt


@dataclass
class TCPForceControllerCfg(TCPControllerCfg):
    gravity: Vector3d = Vector3d().from_xyz([0.0, 0.0, -9.81])


class TCPForceController(TCPController):

    def __init__(self,
                 config: dict[str, Any], 
                 ur_arm: URArm, vrt_ur_arm: VirtualURArm, 
                 control_interface: JointPositionMotorControl | JointVelocityMotorControl, 
                 plug_sensor: PlugSensor
                 ) -> None:
        """ Cartesian tool center point force controller

        Args:
            config: Dictionary to overwrite configuration values
            ur_arm: URArm class reference
            vrt_ur_arm: VirtualURArm class reference
            control_interface: JointPositionMotorControl or JointVelocityMotorControl class reference
            plug_sensor: PlugSensor class reference
        """
        super().__init__(config, ur_arm, vrt_ur_arm, control_interface, plug_sensor)
        # Create configuration and overwrite values
        self.cfg: TCPForceControllerCfg = TCPForceControllerCfg()
        self.cfg.update(**config)

    def update(self, action: npt.NDArray[np.float32]) -> None:
        # Scale action represented in TCP frame
        xyz = Vector3d().from_xyz(action[0:3] * self.cfg.wa_lin)
        rpy = Vector3d().from_xyz(action[3:6] * self.cfg.wa_ang)




