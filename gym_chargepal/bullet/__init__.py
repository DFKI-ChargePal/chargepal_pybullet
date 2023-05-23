""" Configuration file for the bullet classes. """
from math import pi
import numpy as np


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


class BulletDynamicsInfo:
    """
    Indices of the return values of the PyBullet getDynamicsInfo() function.
    """
    MASS = 0
    LATERAL_FRICTION = 1
    LOCAL_INERTIAL_DIAGONAL = 2
    LOCAL_INERTIAL_POS = 3
    LOCAL_INERTIAL_ORI = 4
    RESTITUTION = 5
    ROLLING_FRICTION = 6
    SPINNING_FRICTION = 7
    CONTACT_DAMPING = 8
    CONTACT_STIFFNESS = 9
    BODY_TYPE = 10
    COLLISION_MARGIN = 11


ARM_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


ARM_LINK_NAMES = [
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
]


ARM_JOINT_DEFAULT_VALUES = {
    'shoulder_pan_joint': np.deg2rad(196.57),
    'shoulder_lift_joint': -np.deg2rad(78.87),
    'elbow_joint': np.deg2rad(122.0),
    'wrist_1_joint': -np.deg2rad(43.09),
    'wrist_2_joint': np.deg2rad(106.76),
    'wrist_3_joint': -np.deg2rad(90),
}


ARM_JOINT_LIMITS = {
    'shoulder_pan_joint': (-2.0*pi, 2.0*pi),
    'shoulder_lift_joint': (-2.0*pi, 2.0*pi),
    'elbow_joint': (-2.0*pi, 2.0*pi),
    'wrist_1_joint': (-2.0*pi, 2.0*pi),
    'wrist_2_joint': (-2.0*pi, 2.0*pi),
    'wrist_3_joint': (-2.0*pi, 2.0*pi),
}