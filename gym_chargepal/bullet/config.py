""" Configuration file for the bullet classes. """
import pybullet as p
from numpy import pi

# mypy imports
from typing import Dict, Any


# constants
class BulletLinkState:
    """
    Indices of the return values of the PyBullet getLinkState() function.
    For more information look into the quickstart guide.
    """
    LINK_WORLD_POS = 0  # 3 float tuple
    LINK_WORLD_ORI = 1  # 4 float tuple
    LOCAL_INERTIAL_POS = 2  # 3 float tuple
    LOCAL_INERTIAL_ORI = 3  # 4 float tuple
    WORLD_LINK_FRAME_POS = 4  # 3 float tuple
    WORLD_LINK_FRAME_ORI = 5  # 4 float tuple
    WORLD_LINK_LIN_VEL = 6  # 3 float tuple
    WORLD_LINK_ANG_VEL = 7  # 3 float tuple


class BulletJointInfo:
    """
    Indices of the return values of the PyBullet getJointInfo() function.
    """
    JOINT_INDEX = 0  # 
    JOINT_NAME = 1
    JOINT_TYPE = 2
    Q_INDEX = 3
    U_INDEX = 4
    FLAGS = 5
    JOINT_DAMPING = 6
    JOINT_FRICTION = 7
    JOINT_LOWER_LIMIT = 8
    JOINT_UPPER_LIMIT = 9
    JOINT_MAX_FORCE = 10
    JOINT_MAX_VELOCITY = 11
    LINK_NAME = 12
    JOINT_AXIS = 13
    PARENT_FRAME_POS = 14
    PARENT_FRAME_ORN = 15
    PARENT_INDEX = 16


class BulletJointState:
    """
    Indices of the return values of a single joint state.
    For more information look into the PyBullet quickstart guide -> getJointState()
    """
    JOINT_POSITION = 0
    JOINT_VELOCITY = 1
    JOINT_REACTION_FORCE = 2
    JOINT_MOTOR_TORQUE = 3


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
    'joint_ranges': [2*pi, 2*pi, 2*pi, 2*pi, 2*pi, 2*pi],
    'max_num_iterations': 100,
    'residual_threshold': 1e-5,
}

JACOBIAN: Dict[str, Any] = {}
