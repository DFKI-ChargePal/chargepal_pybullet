""" Configuration file for the sensors classes. """

# mypy
from typing import Dict, Any

from chargepal_pybullet.gym_chargepal.utility.constants import WorldFrame

CONTROLLER: Dict[str, Any] = {}

POSITION_1DOF_CARTESIAN_CONTROLLER = {
    'home_orientation': (1.0, 0.0, 0.0, 0.0),
    'wa': 0.05,  # translation delta in m
    'moving_direction': WorldFrame.Y
}

POSITION_3DOF_CARTESIAN_CONTROLLER = {
    'home_orientation': (1.0, 0.0, 0.0, 0.0),
    'wa': 0.005,  # translation delta in m
}

VELOCITY_3DOF_CARTESIAN_CONTROLLER = {
    'wa': 0.1,  # linear action scale multiplier
}

VELOCITY_6DOF_CARTESIAN_CONTROLLER = {
    'w_lin': 0.1,  # linear action scale multiplier
    'w_ang': 0.05,  # angular action scale multiplier
}
