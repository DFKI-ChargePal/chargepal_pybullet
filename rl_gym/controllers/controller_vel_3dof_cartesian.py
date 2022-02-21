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
from chargepal_pybullet.rl_gym.controllers.config import VELOCITY_3DOF_CARTESIAN_CONTROLLER


LOGGER = logging.getLogger(__name__)


class Velocity3dofCartesianController(Controller):
    """ Continues cartesian XYZ velocity controllers. """
    def __init__(self, hyperparams: Dict[str, Any], jacobian: Jacobian, control_interface: JointVelocityMotorControl, tool_sensor: ToolSensor, joint_sensor: JointSensor):
        config: Dict[str, Any] = copy.deepcopy(VELOCITY_3DOF_CARTESIAN_CONTROLLER)
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
        j_vel = self._joint_sensor.get_vel()
        j_acc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # get Jacobian
        jac_t, jac_r = self._jacobian.calculate(j_pos, j_vel, j_acc)

        jac = np.array(jac_t + jac_r)

        x_dot_com = self._hyperparams['wa'] * action
        x_dot_cur = self._tool_sensor.get_lin_vel()
        x_dot = x_dot_com - x_dot_cur

        x_dot = np.append(x_dot, (0.0, 0.0, 0.0))

        # Multiply the Jacobian Pseudoinverse with the desired cartesian
        # velocities to get the joint velocities
        # q_dot = J^{-1} * x_dot
        q_dot = np.linalg.pinv(jac).dot(x_dot)

        # command robot
        self._controller_interface.update(q_dot)
