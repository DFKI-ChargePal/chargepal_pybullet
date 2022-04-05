""" Configuration file for the sensors classes. """

# mypy
from typing import Dict, Any

from gym_chargepal.utility.constants import WorldFrame

CONTROLLER: Dict[str, Any] = {}

TCP_POSITION_CONTROLLER = {
    'wa_lin': 1.0,  # action scaling in linear directions [m]
    'wa_ang': 1.0,  # action scaling in angular directions [rad]
    'linear_enabled_motion_axis': (True, True, True),
    'angular_enabled_motion_axis': (True, True, True),
    # absolute default postitions for disabled motion directions.
    'plug_lin_config': None,
    'plug_ang_config': None,
}

TCP_VELOCITY_CONTROLLER = {
    'wa_lin': 1.0,  # action scaling in linear directions [m]
    'wa_ang': 1.0,  # action scaling in angular directions [rad]
    'linear_enabled_motion_axis': (True, True, True),
    'angular_enabled_motion_axis': (True, True, True),
}
