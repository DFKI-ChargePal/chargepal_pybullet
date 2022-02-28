import copy
import logging

# mypy
import numpy as np
from typing import Dict, Any, Tuple
from gym_chargepal.sensors.sensor_plug import PlugSensor
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.bullet.joint_position_motor_control import JointPositionMotorControl


from gym_chargepal.controllers.controller import Controller
from gym_chargepal.controllers.config import POSITION_3DOF_CARTESIAN_CONTROLLER


LOGGER = logging.getLogger(__name__)


class Position3dofCartesianController(Controller):
    """ Cartesian XYZ position controller """
    def __init__(
        self, 
        hyperparams: Dict[str, Any], 
        ik_solver: IKSolver, 
        controller_interface: JointPositionMotorControl, 
        plug_sensor: PlugSensor
        ) -> None:
        
        config: Dict[str, Any] = copy.deepcopy(POSITION_3DOF_CARTESIAN_CONTROLLER)
        config.update(hyperparams)
        Controller.__init__(self, config)
        # object references
        self._plug_sensor = plug_sensor
        self._ik_solver = ik_solver
        self._controller_interface = controller_interface
        # constants
        self._ee_q: Tuple[float, ...] = self._hyperparams['home_orientation']

    def update(self, action: np.ndarray) -> None:
        """
        Updates the cartesian position controller
        :param action: 3D continues action [-max_pos ... max_pos]
        :return: None
        """
        # scale action
        d_ee_p = self._hyperparams['wa'] * action
        # update new absolute position
        ee_p = np.array(self._plug_sensor.get_pos())
        next_pose = (tuple(ee_p + d_ee_p), tuple(self._ee_q))
        # transform to joint space positions
        next_j_pos = self._ik_solver.solve(next_pose)
        # command robot
        self._controller_interface.update(next_j_pos)
