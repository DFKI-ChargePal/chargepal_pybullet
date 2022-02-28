import copy
import logging

# mypy
import numpy as np
from typing import Dict, Any, Tuple
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl


from gym_chargepal.controllers.controller import Controller
from gym_chargepal.controllers.config import POSITION_1DOF_CARTESIAN_CONTROLLER


LOGGER = logging.getLogger(__name__)


class Position1dofCartesianController(Controller):
    """ Cartesian XYZ position controller """
    def __init__(
        self, 
        hyperparams: Dict[str, Any], 
        fixed_pos: Tuple[float, ...],
        ik_solver: IKSolver,
        controller_interface: JointPositionMotorControl, 
        plug_sensor: PlugSensor
        ) -> None:

        config: Dict[str, Any] = copy.deepcopy(POSITION_1DOF_CARTESIAN_CONTROLLER)
        config.update(hyperparams)
        Controller.__init__(self, config)
        # object references
        self._plug_sensor = plug_sensor
        self._ik_solver = ik_solver
        self._controller_interface = controller_interface
        # constants
        self._axes: int = self._hyperparams['moving_direction']
        # This parameter is required since only one axis is controlled.
        self._fixed_pos = np.array(fixed_pos, dtype=np.float32)
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
        ee_p = self._fixed_pos
        # update new absolute position in moving direction
        ee_p[self._axes] = self._plug_sensor.get_pos()[self._axes] + action
        next_pose = (tuple(ee_p), tuple(self._ee_q))
        # transform to joint space positions
        next_j_pos = self._ik_solver.solve(next_pose)

        # command robot
        self._controller_interface.update(next_j_pos)
