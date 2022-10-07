""" Configuration file for the evaluation classes. """
# mypy
from typing import Dict, Any

# parent class configuration
EVAL_PTP: Dict[str, Any] = {
    # Task performance criteria
    'task_epsilon_pos': 0.003,  # 3 mm
    'task_epsilon_ang': 1,  # 1 rad
}

# child class configuration
EVAL_PTP_DIST: Dict[str, Any] = {
    'distance_weight': 1.0,
    'distance_exp': 0.4,
}

EVAL_PTP_SPEED: Dict[str, Any] = {
    'distance_weight': 1.0,
    'distance_exp': 0.4,
    'distance_bound': 0.1,
    'speed_weight': 0.1,
    'speed_bound': 0.001,
}

# independent evaluation class configuration
EVAL_DIST: Dict[str, Any] = {
        # Task performance criteria
    'task_epsilon_pos': 0.003,  # 3 mm
    'task_epsilon_ang': 1,  # 1 rad
    'distance_weight': 1.0,
    'distance_exp': 0.4,
}
