# global
import numpy as np

from gym import spaces
from rigmopy import (
    Pose,
    Quaternion,
    Vector3d
)

# local
from gym_chargepal.envs.env_reacher_pos_ctrl import EnvironmentReacherPositionCtrl
from gym_chargepal.envs.env_reacher_vel_ctrl import EnvironmentReacherVelocityCtrl
from gym_chargepal.envs.env_plugger_pos_ctrl import EnvironmentPluggerPositionCtrl


""" Configuration setups point-to-point position controlled environments. """
# Constants
# ROT_PI_2_X = Quaternion().from_euler_angle(angles=(np.pi/2, 0.0, 0.0))

# ROT_PI_X = Quaternion().from_euler_angle(angles=(np.pi, 0.0, 0.0))

# DEFAULT_TARGET_POS = Vector3d().from_xyz((0.0, 0.0, 1.2))

# ///////////////////////////////////////////////////// #
# ///                                               /// #
# ///   Environments with TCP position controller   /// #
# ///                                               /// #
# ///////////////////////////////////////////////////// #
# reacher_1dof_position_ctrl_v0 = {
    
#     'environment': {
#         'type': EnvironmentReacherPositionCtrl,
#         'T': 100,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.2, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.0, 0.025, 0.0), (0.0, 0.0, 0.0)),
#     },

#     'low_level_control': {
#         'wa_lin': 0.025,  # action scaling in linear directions [m]
#         'linear_enabled_motion_axis': (False, True, False),
#         'angular_enabled_motion_axis': (False, False, False),
#     }
# }


# reacher_3dof_position_ctrl_v0 = {

#     'environment': {
#         'type': EnvironmentReacherPositionCtrl,
#         'T': 100,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
#         # Target configuration
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.2, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.01, 0.05, 0.01), (0.0, 0.0, 0.0)),
#         },

#     'low_level_control': {
#         'wa_lin': 0.025,  # action scaling in linear directions [m]
#         'linear_enabled_motion_axis': (True, True, True),
#         'angular_enabled_motion_axis': (False, False, False),
#         },
# }


# reacher_6dof_position_ctrl_v0 = {

#     'environment': {
#         'type': EnvironmentReacherPositionCtrl,
#         'T': 100,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.2, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.05, 0.05, 0.05), (0.1, 0.1, 0.1)),
#         },

#     'low_level_control': {
#         'wa_lin': 0.025,  # action scaling in linear directions [m]
#         'wa_ang': 0.05,  # action scaling in angular directions [rad]
#         },
# }


testbed_reacher_3dof_position_ctrl_v1 = {

    'environment': {
        'type': EnvironmentReacherPositionCtrl,
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration with respect to the arm base
        'target_config': Pose().from_xyz((0.75, 0.1, 0.4)).from_euler_angle((0.0, 90.0, 0.0), degrees=True),
        # Start configuration relative to target configuration'
        'start_config': Pose().from_xyz((0.15, 0.3, -0.15)),
        # Reset variance of the start pose
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        'reset_variance': ((0.1, 0.1, 0.1), (0.0, 0.0, 0.0)),
        },

    'low_level_control': {
        'wa_lin': 0.05,  # action scaling in linear directions [m]
        'linear_enabled_motion_axis': (True, True, True),
        'angular_enabled_motion_axis': (False, False, False),
        },

}

testbed_reacher_6dof_position_ctrl_v1 = {

    'environment': {
        'type': EnvironmentReacherPositionCtrl,
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration with respect to the arm base
        'target_config': Pose().from_xyz((0.75, 0.1, 0.4)).from_euler_angle((0.0, 90.0, 0.0), degrees=True),
        # Start configuration relative to target configuration'
        'start_config': Pose().from_xyz((0.15, 0.3, -0.15)),
        # Reset variance of the start pose
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        'reset_variance': ((0.1, 0.1, 0.1), (0.0, 0.0, 0.0)),
        },

    'low_level_control': {
        'wa_lin': 0.05,  # action scaling in linear directions [m]
        'wa_ang': 0.05 * np.pi,  # action scaling in angular directions [rad]
        },

}



# testbed_reacher_6dof_position_ctrl_v0 = {

#     'environment': {
#         'type': EnvironmentReacherPositionCtrl,
#         'T': 100,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_xyz((0.5, 1.0, 0.4)).from_euler_angle((-np.pi/2, 0.0, 0.0)),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((-0.15, -0.3, -0.175)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.1, 0.1, 0.1), (0.05, 0.05, 0.05)),
#         },

#     'world': {
#         'robot_urdf': 'chargepal_testbed_cic.urdf',
#         'socket_urdf': 'tdt_socket.urdf',
#         'cam_distance': 0.75,
#         'cam_yaw': 105,
#         'cam_pitch': -15.0,
#         'cam_x': 0.8,
#         'cam_y': 0.8,
#         'cam_z': 0.15,
#         'plane_config': Pose().from_xyz((0.0, 0.0, -0.8136)),
#         'robot_config': Pose(),
#         },

    # 'ur_arm': {
    #     'joint_default_values': {
    #         'shoulder_pan_joint': np.deg2rad(196.57),
    #         'shoulder_lift_joint': -np.deg2rad(78.87),
    #         'elbow_joint': np.deg2rad(122.0),
    #         'wrist_1_joint': -np.deg2rad(43.09),
    #         'wrist_2_joint': np.deg2rad(106.76),
    #         'wrist_3_joint': -np.deg2rad(90),
    #         },
    #     },
    #     # np.deg2rad((196.57, -78.87, 122.0, -43.09, 106.76, -90)).tolist()

#     'low_level_control': {
#         'wa_lin': 0.05,  # action scaling in linear directions [m]
#         'wa_ang': 0.05 * np.pi,  # 
#         },

# }


# plugger_position_ctrl_v0 = {

#     'environment': {
#         'type': EnvironmentPluggerPositionCtrl,
#         'T': 200,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.1, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.01, 0.005, 0.01), (0.05, 0.05, 0.05)),
#         },
# }


# SOCKET_POS = Vector3d().from_xyz((0.0, -0.1/2.0, 0.0))
# SOCKET_ORI = Quaternion().from_euler_angle((0.0, 0.0, np.pi))
# ROD_DIAMETER = "32d5"
# plugger_position_ctrl_v1 = {

#     'environment': {
#         'type': EnvironmentPluggerPositionCtrl,
#         'T': 200,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.075, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.01, 0.005, 0.01), (0.05, 0.05, 0.05)),
#         # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
#         },

#     'world': {
#         'robot_urdf': f'cpm_fix_rod_{ROD_DIAMETER}.urdf',
#         'socket_urdf': 'adapter_station_square_socket.urdf',
#         'socket_config': Pose().from_pq(SOCKET_POS, SOCKET_ORI),
#         },
# }


# ROD_DIAMETER = "34"
# plugger_position_ctrl_v2 = {
#     'environment': {
#         'type': EnvironmentPluggerPositionCtrl,
#         'T': 200,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.075, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.01, 0.005, 0.01), (0.05, 0.05, 0.05)),
#         # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
#         },

#     'world': {
#         'robot_urdf': f'cpm_fix_rod_{ROD_DIAMETER}.urdf',
#         'socket_urdf': 'adapter_station_square_socket.urdf',
#         'socket_config': Pose().from_pq(SOCKET_POS, SOCKET_ORI),
#         },
# }


# testbed_plugger_position_ctrl_v0 = {

#     'environment': {
#         'type': EnvironmentPluggerPositionCtrl,
#         'T': 200,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_xyz((0.5, 0.8, 0.0)).from_pq(q=ROT_PI_X),
#         # Start configuration relative to target configuration
#         'start_config': Pose().from_xyz((0.0, 0.0, 0.02+0.095)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.01, 0.01, 0.001), (0.05, 0.05, 0.05)),
#         # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
#         },

#     'world': {
#         'robot_urdf': 'chargepal_testbed_tdt.urdf',
#         'socket_urdf': 'tdt_socket.urdf',
#         'cam_distance': 0.75,
#         'cam_yaw': 105,
#         'cam_pitch': -15.0,
#         'cam_x': 0.8,
#         'cam_y': 0.8,
#         'cam_z': 0.15,
#         'plane_config': Pose().from_xyz((0.0, 0.0, -0.8136)),
#         'robot_config': Pose(),
#         'socket_config': Pose().from_xyz((0.5, 0.8, 0.0)),
#         'gui_txt': "",
#         'gui_txt_pos': (0.75, 1.25, 0.6),
#         },

#     'ur_arm': {
#         'joint_default_values': {
#             'shoulder_pan_joint': np.pi,
#             'shoulder_lift_joint': -7*np.pi/36,
#             'elbow_joint': -np.pi/2 - 7*np.pi/36,
#             'wrist_1_joint': -np.pi/2 - 4*np.pi/36,
#             'wrist_2_joint': np.pi/2,
#             'wrist_3_joint': -np.pi/2,
#             },
#         },

#     'ft_sensor': {
#         'render_bar': False,
#         },

#     'socket_sensor': {
#         'pos_noise': (0.005, 0.005, 0.010),
#         'pos_bias': (0.003, -0.02, 0.001),
#         'ori_noise': (2*np.pi*0.001, 2*np.pi*0.001, 2*np.pi*0.001),
#         'ori_bias': (-0.005, 0.001, -0.0025),
#         },
# }


# ///////////////////////////////////////////////////// #
# ///                                               /// #
# ///   Environments with TCP velocity controller   /// #
# ///                                               /// #
# ///////////////////////////////////////////////////// #
# reacher_3dof_velocity_ctrl_v0 = {
#     'environment': {
#         'type': EnvironmentReacherVelocityCtrl,
#         'T': 100,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.2, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.01, 0.05, 0.01), (0.0, 0.0, 0.0)),
#         },

#     'low_level_control': {
#         'linear_enabled_motion_axis': (True, True, True),
#         'angular_enabled_motion_axis': (False, False, False),
#         },
# }

# reacher_6dof_velocity_ctrl_v0 = {
#     'environment': {
#         'type': EnvironmentReacherVelocityCtrl,
#         'T': 100,
#         'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
#         'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
#         # Target configuration
#         'target_config': Pose().from_pq(DEFAULT_TARGET_POS, ROT_PI_2_X),
#         # Start configuration relative to target configuration'
#         'start_config': Pose().from_xyz((0.0, 0.2, 0.0)),
#         # Reset variance of the start pose
#         'reset_variance': ((0.05, 0.05, 0.05), (0.1, 0.1, 0.1)),
#         },
# }
