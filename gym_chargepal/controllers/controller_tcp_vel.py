# global
import copy
import logging
import numpy as np
from dataclasses import dataclass

# local
from gym_chargepal.utility.constants import MotionAxis
from gym_chargepal.controllers.controller import Controller, ControllerCfg

# mypy
import numpy.typing as npt
from typing import Any, Dict, List, Tuple
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.sensors.sensor_joints import JointSensor
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

LOGGER = logging.getLogger(__name__)


@dataclass
class TcpVelocityControllerCfg(ControllerCfg):
    linear_enabled_motion_axis: Tuple[bool, ...] = (True, True, True)
    angular_enabled_motion_axis: Tuple[bool, ...] = (True, True, True)
    wa_lin: float = 0.2  # action scaling in linear directions [m]
    wa_ang: float = 0.2 * np.pi  # action scaling in angular directions [rad]


class TcpVelocityController(Controller):
    """ Cartesian tool center point velocity controller """
    def __init__(self,
        config: Dict[str, Any],
        jacobian: Jacobian,
        control_interface: JointVelocityMotorControl,
        plug_sensor: PlugSensor,
        joint_sensor: JointSensor
        ) -> None:
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: TcpVelocityControllerCfg = TcpVelocityControllerCfg()
        self.cfg.update(**config)
        # object references
        self.jacobian = jacobian
        self.control_interface = control_interface
        self.plug_sensor = plug_sensor
        self.joint_sensor = joint_sensor
        # constants
        self.wa_lin = self.cfg.wa_lin
        self.wa_ang = self.cfg.wa_ang
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
        Updates the tcp velocity controller
        : param action: Action array; The action sequence is defined as (x y z roll pitch yaw).
                        If not all motion directions are enabled, the actions will be executed
                        in the order in which they are given.
        : return: None
        """
        # get joint configuration
        j_pos = self.joint_sensor.noisy_pos
        j_vel = self.joint_sensor.noisy_vel
        j_acc = tuple([0.0] * 6)

        # get Jacobians
        jac_t, jac_r = self.jacobian.calculate(j_pos, j_vel, j_acc)
        # merge into one jacobian matrix
        jac = np.array(jac_t + jac_r)

        # invert the jacobian to map from tcp velocities to joint velocities
        # be careful of singularities and non square matrices
        # use pseudo-inverse when this is the case
        # this is all the time for 7 dof arms like panda
        if jac.shape[1] > np.linalg.matrix_rank(jac.T):
            inv_jac = np.linalg.pinv(jac)
        else:
            inv_jac = np.linalg.inv(jac)

        # scale actions
        action[self.lin_action_ids] *= self.wa_lin
        action[self.ang_action_ids] *= self.wa_ang

        # Get current tcp velocities
        tcp_dot_twist = self.plug_sensor.noisy_V_wrt_world.to_numpy()
        # Separate into linear and angular part
        tcp_dot_lin = tcp_dot_twist[0:3]
        tcp_dot_ang = tcp_dot_twist[3:6]

        # Calculate difference of commanded tcp velocity and current tcp velocity 
        # to get new desired absolute velocity.
        # Degrees of freedom that are not controlled are set to zero.
        tcp_dot_lin_ = np.zeros(3)
        tcp_dot_ang_ = np.zeros(3)
        tcp_dot_lin_[self.lin_motion_axis[MotionAxis.ENABLED]] = action[self.lin_action_ids] - tcp_dot_lin[self.lin_motion_axis[MotionAxis.ENABLED]]
        tcp_dot_ang_[self.ang_motion_axis[MotionAxis.ENABLED]] = action[self.ang_action_ids] - tcp_dot_ang[self.ang_motion_axis[MotionAxis.ENABLED]]
        tcp_dot = np.concatenate([tcp_dot_lin_, tcp_dot_ang_])

        # convert desired velocities from cart space to joint space
        q_dot = np.matmul(inv_jac, tcp_dot)

        # send command to robot
        self.control_interface.update(q_dot)
