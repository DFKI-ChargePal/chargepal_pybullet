""" Configuration file for the bullet classes. """
import pybullet as p
from numpy import pi

# mypy imports
from typing import Dict, Any

# configurations
JOINT_POSITION_MOTOR_CONTROL = {
    'control_mode': p.POSITION_CONTROL,
    'target_vel': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'forces': [330.0, 330.0, 150.0, 54.0, 54.0, 54.0],
    'pos_gains': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'vel_gains': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

JOINT_VELOCITY_MOTOR_CONTROL = {
    'control_mode': p.VELOCITY_CONTROL,
    'target_pos': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'target_vel': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'forces': [330.0, 330.0, 150.0, 54.0, 54.0, 54.0],
    'pos_gains': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'vel_gains': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

IK_SOLVER = {
    'lower_limits': [-pi, -pi, -pi, -pi, -pi, -pi],
    'upper_limits': [pi, pi, pi, pi, pi, pi],
    'joint_ranges': [2 * pi, 2 * pi, 2 * pi, 2 * pi, 2 * pi, 2 * pi],
    'max_num_iterations': 100,
    'residual_threshold': 1e-5,
}

JACOBIAN: Dict[str, Any] = {}
