from __future__ import annotations

# global
import numpy as np
from dataclasses import dataclass
from rigmopy import Vector3d, Vector6d

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
        # Scale action represented in sensor frame
        f_add_sensor = Vector3d().from_xyz(action[0:3] * self.cfg.wa_lin)
        t_add_sensor = Vector3d().from_xyz(action[3:6] * self.cfg.wa_ang)

        # Rotate action wrench in arm base frame.
        q_ft2arm = self.ur_arm.q_arm2fts.inverse()
        f_add_arm = q_ft2arm.apply(f_add_sensor)
        t_add_arm = q_ft2arm.apply(t_add_sensor)
        ft_add_arm = Vector6d().from_Vector3d(f_add_arm, t_add_arm)

        # Get latest force readings in arm base frame
        ft_old_sensor = self.ur_arm.raw_wrench.split()
        f_old_arm = q_ft2arm.apply(ft_old_sensor[0])
        t_old_arm = q_ft2arm.apply(ft_old_sensor[1])

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
        t_plug_weight_wrt_arm = Vector3d().from_xyz(np.cross(f_plug_weight_wrt_arm.xyz, p_sensor2com_wrt_arm.xyz))

        # Add compensation to current sensor readings
        f_old_arm_comp = f_old_arm - f_sensor_weight_wrt_arm - f_plug_weight_wrt_arm
        t_old_arm_comp = t_old_arm - t_sensor_weight_wrt_arm - t_plug_weight_wrt_arm

        # Merge force torque signal
        if all([_ == 0.0 for _ in ft_old_sensor[0].xyz]) and all([_ == 0.0 for _ in ft_old_sensor[1].xyz]):
            ft_old_arm_comp = Vector6d()
        else:
            ft_old_arm_comp = Vector6d().from_Vector3d(f_old_arm_comp, t_old_arm_comp)

        # New force torque signal
        f_net = ft_add_arm - ft_old_arm_comp
        f_ctrl = self.spatial_pd_ctrl.update(f_net, self.cfg.period)
        self._to_joint_commands(f_ctrl)
