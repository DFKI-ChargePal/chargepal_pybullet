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
        self._ik_solver = ik_solver
        self._controller_interface = controller_interface
        # constants
        self._wa_lin = self.cfg.wa_lin
        self._wa_ang = self.cfg.wa_ang
        assert self.cfg.plug_lin_config
        assert self.cfg.plug_ang_config
        self._plug_lin_config = np.array(self.cfg.plug_lin_config)
        self._plug_ang_config = np.array(self.cfg.plug_ang_config)
        # mapping of the enabled motion axis to the indices
        self._lin_motion_axis: Dict[bool, List[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.linear_enabled_motion_axis):
            self._lin_motion_axis[mode].append(axis)
        self._ang_motion_axis: Dict[bool, List[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self.cfg.angular_enabled_motion_axis):
            self._ang_motion_axis[mode].append(axis)
        # Slices for the linear and angular actions.
        start_idx = 0
        stop_idx = len(self._lin_motion_axis[MotionAxis.ENABLED])
        self._lin_action_ids = slice(start_idx, stop_idx)
        start_idx = stop_idx
        stop_idx = start_idx + len(self._ang_motion_axis[MotionAxis.ENABLED])
        self._ang_action_ids = slice(start_idx, stop_idx)


    def update(self, action: npt.NDArray[np.float32]) -> None:
        """
        Updates the tcp position controller
        : param action: Action array; The action sequence is defined as (x y z roll pitch yaw).
                        If not all motion directions are enabled, the actions will be executed 
                        in the order in which they are given.
        : return: None
        """
        # Scale action
        action[self._lin_action_ids] *= self._wa_lin
        action[self._ang_action_ids] *= self._wa_ang
        # Get current pose
        plug_lin_pos = np.array(self.plug_sensor.get_pos())
        plug_ang_pos = np.array(p.getEulerFromQuaternion(self.plug_sensor.get_ori()))
        # Increment pose by action
        plug_lin_pos[self._lin_motion_axis[MotionAxis.ENABLED]] += action[self._lin_action_ids]
        plug_ang_pos[self._ang_motion_axis[MotionAxis.ENABLED]] += action[self._ang_action_ids]
        # Set disabled axis to default values to avoid pos drift.
        plug_lin_pos[self._lin_motion_axis[MotionAxis.DISABLED]] = self._plug_lin_config[
            self._lin_motion_axis[MotionAxis.DISABLED]
            ]
        plug_ang_pos[self._ang_motion_axis[MotionAxis.DISABLED]] = self._plug_ang_config[
            self._ang_motion_axis[MotionAxis.DISABLED]
            ]
        # Compose new plug pose
        plug_pose = (tuple(plug_lin_pos), tuple(p.getQuaternionFromEuler(plug_ang_pos)))
        # Transform to joint space positions
        joint_pos = self._ik_solver.solve(plug_pose)
        # Send command to robot
        self._controller_interface.update(joint_pos)
