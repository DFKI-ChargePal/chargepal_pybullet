# global
import logging
import numpy as np
import pybullet as p
from dataclasses import dataclass

# local
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.utility.constants import MotionAxis
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.controllers.controller import Controller, ControllerCfg
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl

# mypy
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class TcpPositionControllerCfg(ControllerCfg):
    linear_enabled_motion_axis: Tuple[bool, ...] = (True, True, True)
    angular_enabled_motion_axis: Tuple[bool, ...] = (True, True, True)
    # Absolute default positions for disabled motion directions
    plug_lin_config: Optional[Tuple[float, ...]] = None
    plug_ang_config: Optional[Tuple[float, ...]] = None
    # Action scaling
    wa_lin: float = 0.01  # action scaling in linear directions [m]
    wa_ang: float = 0.01 * np.pi  # action scaling in angular directions [rad]


class TcpPositionController(Controller):
    """ Cartesian tool center point position controller """
    def __init__(self,
        config: Dict[str, Any],
        ik_solver: IKSolver,
        controller_interface: JointPositionMotorControl,
        plug_sensor: PlugSensor
        ) -> None:
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: TcpPositionControllerCfg = TcpPositionControllerCfg()
        self.cfg.update(**config)
        # object references
        self.plug_sensor = plug_sensor
        self.ik_solver = ik_solver
        self.controller_interface = controller_interface
        # constants
        self.wa_lin = self.cfg.wa_lin
        self.wa_ang = self.cfg.wa_ang
        assert self.cfg.plug_lin_config
        assert self.cfg.plug_ang_config
        self.plug_lin_config = np.array(self.cfg.plug_lin_config)
        self.plug_ang_config = np.array(self.cfg.plug_ang_config)
        # mapping of the enabled motion axis to the indices
        self.lin_motion_axis: Dict[bool, List[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.linear_enabled_motion_axis):
            self.lin_motion_axis[mode].append(axis)
        self.ang_motion_axis: Dict[bool, List[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.angular_enabled_motion_axis):
            self.ang_motion_axis[mode].append(axis)
        # Slices for the linear and angular actions.
        start_idx = 0
        stop_idx = len(self.lin_motion_axis[MotionAxis.ENABLED])
        self.lin_action_ids = slice(start_idx, stop_idx)
        start_idx = stop_idx
        stop_idx = start_idx + len(self.ang_motion_axis[MotionAxis.ENABLED])
        self.ang_action_ids = slice(start_idx, stop_idx)

    def update(self, action: npt.NDArray[np.float32]) -> None:
        """
        Updates the tcp position controller
        : param action: Action array; The action sequence is defined as (x y z roll pitch yaw).
                        If not all motion directions are enabled, the actions will be executed 
                        in the order in which they are given.
        : return: None
        """
        # Scale action
        action[self.lin_action_ids] *= self.wa_lin
        action[self.ang_action_ids] *= self.wa_ang
        # Get current pose
        plug_lin_pos = self.plug_sensor.get_pos().xyz1[0:3]
        plug_ang_pos = np.array(p.getEulerFromQuaternion(self.plug_sensor.get_ori().xyzw))
        # Increment pose by action
        plug_lin_pos[self.lin_motion_axis[MotionAxis.ENABLED]] += action[self.lin_action_ids]
        plug_ang_pos[self.ang_motion_axis[MotionAxis.ENABLED]] += action[self.ang_action_ids]
        # Set disabled axis to default values to avoid pos drift.
        plug_lin_pos[self.lin_motion_axis[MotionAxis.DISABLED]] = self.plug_lin_config[
            self.lin_motion_axis[MotionAxis.DISABLED]
            ]
        plug_ang_pos[self.ang_motion_axis[MotionAxis.DISABLED]] = self.plug_ang_config[
            self.ang_motion_axis[MotionAxis.DISABLED]
            ]
        # Compose new plug pose
        plug_pose = (tuple(plug_lin_pos), tuple(p.getQuaternionFromEuler(plug_ang_pos)))
        # Transform to joint space positions
        joint_pos = self.ik_solver.solve(plug_pose)
        # Send command to robot
        self.controller_interface.update(joint_pos)
