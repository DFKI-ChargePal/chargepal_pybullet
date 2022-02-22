""" This file defines the default configuration of the pybullet worlds. """
import numpy as np


ur_joint_idx = {
    'SHOULDER_PAN': 1,
    'SHOULDER_LIFT': 2,
    'ELBOW_JOINT': 3,
    'WRIST_1': 4,
    'WRIST_2': 5,
    'WRIST_3': 6,
}

ur_joint_x0 = {
    'SHOULDER_PAN': np.pi,
    'SHOULDER_LIFT': -1.0,
    'ELBOW_JOINT': 2.171,
    'WRIST_1': -1.171,
    'WRIST_2': 1.571,
    'WRIST_3': 0.0,
}


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


class BulletJointState:
    """
    Indices of the return values of a single joint state.
    For more information look into the PyBullet quickstart guide -> getJointState()
    """
    JOINT_POSITION = 0
    JOINT_VELOCITY = 1
    JOINT_REACTION_FORCE = 2
    JOINT_MOTOR_TORQUE = 3


WORLD = {
    'hz': 240,  # frequency physic engine !not recommended to changing this value!
    'gravity': (0, 0, -9.81),
    'joint_idx': ur_joint_idx,
    'joint_x0': ur_joint_x0,
    'link_state_idx': BulletLinkState,
    'joint_state_idx': BulletJointState,
}

WORLD_PTP = {
    'robot_urdf': 'gym_env/assets/ur_e_description/urdf/ur10e/ur10e_robot.urdf',
    'platform_urdf': 'gym_env/assets/platform/urdf/platform.urdf',
    'tool_frame_idx': 11,
    'target': (0.0, 0.9, 1.0),
}

WORLD_PIH = {
    'arm': 'gym_env/assets/ur_e_description/urdf/ur10e/ur10e_robot.urdf',
    'platform': 'gym_env/assets/platform/urdf/platform.urdf',
    'pillar': 'gym_env/assets/pillar/urdf/pillar.urdf',
    'tool_frame_idx': 11,
    'ft_sensor_idx': 8,
    'target_frame_idx': 4,
    'virtual_frame_idx': {
        'tool': (12, 13, 14),
        'target': (5, 6, 7),
    },
}
