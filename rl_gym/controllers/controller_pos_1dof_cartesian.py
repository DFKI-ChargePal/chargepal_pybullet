import copy
import logging

# mypy
import numpy as np
from typing import Dict, Any, Tuple
from gym_env.sensors.sensor_tool import ToolSensor
from gym_env.bullet.ik_solver import IKSolver
from gym_env.bullet.joint_position_motor_control import JointPositionMotorControl


from gym_env.controllers.controller import Controller
from gym_env.controllers.config import POSITION_1DOF_CARTESIAN_CONTROLLER


LOGGER = logging.getLogger(__name__)


class Position1dofCartesianController(Controller):
    """ Cartesian XYZ position controller """
    def __init__(self, hyperparams: Dict[str, Any], ik_solver: IKSolver, controller_interface: JointPositionMotorControl, tool_sensor: ToolSensor):
        config: Dict[str, Any] = copy.deepcopy(POSITION_1DOF_CARTESIAN_CONTROLLER)
        config.update(hyperparams)
        Controller.__init__(self, config)
        # object references
        self._tool_sensor = tool_sensor
        self._ik_solver = ik_solver
        self._controller_interface = controller_interface
        # constants
        self._axes: int = self._hyperparams['moving_direction']
        self._tgt = np.array(self._hyperparams['target'], dtype=np.float32)
        self._ee_q: Tuple[float, ...] = self._hyperparams['home_orientation']

    def update(self, action: np.ndarray) -> None:
        """
        Updates the cartesian position controller
        :param action: scalar continues action [-max_pos ... max_pos]
        :return: None
        """
        # scale action
        action = self._hyperparams['wa'] * action
        # use target to avoid drift in non-control directions
        ee_p = self._tgt
        # update new absolute position in moving direction
        ee_p[self._axes] = self._tool_sensor.get_pos()[self._axes] + action
        next_pose = (tuple(ee_p), tuple(self._ee_q))
        # transform to joint space positions
        next_j_pos = self._ik_solver.solve(next_pose)
        # command robot
        self._controller_interface.update(next_j_pos)
