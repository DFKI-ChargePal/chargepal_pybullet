""" Configuration setups point-to-point position controlled environments. """
# global
import numpy as np
from gym import spaces
from rigmopy import Pose

# local
from gym_chargepal.envs.env_reacher_pos_ctrl import EnvironmentReacherPositionCtrl
from gym_chargepal.envs.env_plugger_pos_ctrl import EnvironmentPluggerPositionCtrl

# ///////////////////////////////////////////////////////////// #
# ///                                                       /// #
# ///   Environments Reacher with TCP position controller   /// #
# ///                                                       /// #
# ///////////////////////////////////////////////////////////// #

testbed_reacher_3dof_position_ctrl_v1 = {

    'environment': {
        'type': EnvironmentReacherPositionCtrl,
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Start configuration relative to target configuration'
        'start_config': Pose().from_xyz((0.15, 0.3, -0.15)),
        # Reset variance of the start pose
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        'reset_variance': ((0.1, 0.1, 0.1), (0.0, 0.0, 0.0)),
    },

    'target': {
        # Virtual target configuration w.r.t. the arm base
        'X_arm2tgt': Pose().from_xyz((0.75, 0.1, 0.4)).from_euler_angle((0.0, 90.0, 0.0), degrees=True), 
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
        # Start configuration relative to target configuration'
        'start_config': Pose().from_xyz((0.15, 0.3, -0.15)),
        # Reset variance of the start pose
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        'reset_variance': ((0.1, 0.1, 0.1), (0.1 * np.pi, 0.1 * np.pi, 0.1 * np.pi)),
    },

    'target': {
        # Virtual target configuration w.r.t. the arm base
        'X_arm2tgt': Pose().from_xyz((0.75, 0.1, 0.4)).from_euler_angle((0.0, 90.0, 0.0), degrees=True), 
    },

    'low_level_control': {
        'wa_lin': 0.05,  # action scaling in linear directions [m]
        'wa_ang': 0.05 * np.pi,  # action scaling in angular directions [rad]
    },

}



# ///////////////////////////////////////////////////////////// #
# ///                                                       /// #
# ///   Environments Plugger with TCP position controller   /// #
# ///                                                       /// #
# ///////////////////////////////////////////////////////////// #

testbed_plugger_6dof_position_ctrl_v1 = {

    'environment': {
        'type': EnvironmentPluggerPositionCtrl,
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Start configuration relative to target configuration'
        # 'start_config': Pose(),
        'start_config': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        # 'reset_variance': ((0.1, 0.1, 0.1), (0.1 * np.pi, 0.1 * np.pi, 0.1 * np.pi)),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, -np.pi/2, 0.0)),
    },

    'low_level_control': {
        'wa_lin': 0.05,          # action scaling in linear directions [m]
        'wa_ang': 0.05 * np.pi,  # action scaling in angular directions [rad]
    },
}