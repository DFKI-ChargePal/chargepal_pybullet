import logging
import copy

import numpy as np

# mypy
from typing import Dict, Any
from chargepal_pybullet.rl_gym.bullet.jacobian import Jacobian
from chargepal_pybullet.rl_gym.bullet.joint_velocity_motor_control import JointVelocityMotorControl
from chargepal_pybullet.rl_gym.sensors.sensor_tool import ToolSensor
from chargepal_pybullet.rl_gym.sensors.sensor_joints import JointSensor


from chargepal_pybullet.rl_gym.controllers.controller import Controller
from chargepal_pybullet.rl_gym.controllers.config import VELOCITY_6DOF_CARTESIAN_CONTROLLER


LOGGER = logging.getLogger(__name__)


class Velocity6dofCartesianControllerC(Controller):
    """ Continues cartesian XYZ velocity controllers. """
    def __init__(self, hyperparams: Dict[str, Any], jacobian: Jacobian, control_interface: JointVelocityMotorControl, tool_sensor: ToolSensor, joint_sensor: JointSensor):
        config: Dict[str, Any] = copy.deepcopy(VELOCITY_6DOF_CARTESIAN_CONTROLLER)
        config.update(hyperparams)
        Controller.__init__(self, config)
        # object references
        self._jacobian = jacobian
        self._joint_sensor = joint_sensor
        self._tool_sensor = tool_sensor
        self._controller_interface = control_interface

    def update(self, action: np.ndarray) -> None:
        """
        Updates the cartesian velocity controllers
        :param action: 3D continues action [-max_vel ... max_vel]
        :return: None
        """
        j_pos = self._joint_sensor.get_pos()
        j_vel = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        j_acc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # get Jacobian
        jac_t, jac_r = self._jacobian.calculate(j_pos, j_vel, j_acc)

        jac = np.array(jac_t + jac_r)

        xd_lin_com = self._hyperparams['w_lin'] * action[:3]
        xd_ang_com = self._hyperparams['w_ang'] * np.pi * action[3:]
        xd_lin_cur = self._tool_sensor.get_lin_vel()
        xd_ang_cur = self._tool_sensor.get_ang_vel()
        xd_lin = xd_lin_com - xd_lin_cur
        xd_ang = xd_ang_com - xd_ang_cur

        x_dot = np.append(xd_lin, xd_ang)

        # Multiply the Jacobian Pseudoinverse with the desired cartesian
        # velocities to get the joint velocities
        # q_dot = J^{-1} * x_dot
        q_dot = np.linalg.pinv(jac).dot(x_dot)

        # command robot
        self._controller_interface.update(q_dot)
