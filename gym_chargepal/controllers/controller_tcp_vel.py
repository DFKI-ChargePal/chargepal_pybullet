import copy
import logging

import numpy as np
import pybullet as p

from gym_chargepal.utility.constants import MotionAxis
from gym_chargepal.controllers.controller import Controller
from gym_chargepal.controllers.config import TCP_VELOCITY_CONTROLLER

# mypy
import numpy.typing as npt
from typing import Any, Dict, List
from gym_chargepal.bullet.jacobian import Jacobian
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.sensors.sensor_joints import JointSensor
from gym_chargepal.bullet.joint_velocity_motor_control import JointVelocityMotorControl

LOGGER = logging.getLogger(__name__)


class TcpVelocityController(Controller):
    """ Cartesian tool center point velocity controller """
    def __init__(self,
        hyperparams: Dict[str, Any],
        jacobian: Jacobian,
        control_interface: JointVelocityMotorControl,
        plug_sensor: PlugSensor,
        joint_sensor: JointSensor
        ) -> None:
        # parameter update
        config: Dict[str, Any] = copy.deepcopy(TCP_VELOCITY_CONTROLLER)
        config.update(hyperparams)
        Controller.__init__(self, config)
        # object references
        self._jacobian = jacobian
        self._control_interface = control_interface
        self._plug_sensor = plug_sensor
        self._joint_sensor = joint_sensor
        # constants
        self._wa_lin: float = self._hyperparams['wa_lin']
        self._wa_ang: float = self._hyperparams['wa_ang']
        # mapping of the enabled motion axis to the indices
        self._lin_motion_axis: Dict[bool, List[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self._hyperparams['linear_enabled_motion_axis']):
            self._lin_motion_axis[mode].append(axis)
        self._ang_motion_axis: Dict[bool, List[int]] = {
            MotionAxis.ENABLED: [], 
            MotionAxis.DISABLED: [],
            }
        for axis, mode in enumerate(self._hyperparams['angular_enabled_motion_axis']):
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
        Updates the tcp velocity controller
        : param action: Action array; The action sequence is defined as (x y z roll pitch yaw).
                        If not all motion directions are enabled, the actions will be executed
                        in the order in which they are given.
        : return: None
        """
        # get joint configuration
        j_pos = self._joint_sensor.get_pos()
        j_vel = self._joint_sensor.get_vel()
        j_acc = tuple([0.0] * 6)

        # get Jacobians
        jac_t, jac_r = self._jacobian.calculate(j_pos, j_vel, j_acc)
        # merge into one jacobian matrix
        jac = np.array(jac_t + jac_r)

        # invert the jacobian to map from tcp velocities to joint velocities
        # be careful of singnularities and non square matrices
        # use pseudo-inverse when this is the case
        # this is all the time for 7 dof arms like panda
        if jac.shape[1] > np.linalg.matrix_rank(jac.T):
            inv_jac = np.linalg.pinv(jac)
        else:
            inv_jac = np.linalg.inv(jac)

        # scale actions
        action[self._lin_action_ids] *= self._wa_lin
        action[self._ang_action_ids] *= self._wa_ang

        tcp_dot_lin = np.array(self._plug_sensor.get_lin_vel())
        tcp_dot_ang = np.array(self._plug_sensor.get_ang_vel())

        # Calculate difference of commanded tcp velocity and current tcp velocity 
        # to get new disired absolute velocity.
        # Degrees of freedom that are not controlled are set to zero.
        tcp_dot_lin_ = np.zeros(3)
        tcp_dot_ang_ = np.zeros(3)
        tcp_dot_lin_[self._lin_motion_axis[MotionAxis.ENABLED]] = action[self._lin_action_ids] - tcp_dot_lin[self._lin_motion_axis[MotionAxis.ENABLED]]
        tcp_dot_ang_[self._ang_motion_axis[MotionAxis.ENABLED]] = action[self._ang_action_ids] - tcp_dot_ang[self._ang_motion_axis[MotionAxis.ENABLED]]
        tcp_dot = np.concatenate([tcp_dot_lin_, tcp_dot_ang_])

        # convert desired velocities from cart space to joint space
        q_dot = np.matmul(inv_jac, tcp_dot)

        # send command to robot
        self._control_interface.update(q_dot)
