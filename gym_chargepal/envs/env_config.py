""" Configuration setups point-to-point position controlled environments. """
# global
import numpy as np
from gym import spaces
from rigmopy import Pose

# local
from gym_chargepal.envs.env_plugger_frc_ctrl import EnvironmentPluggerForceCtrl
from gym_chargepal.envs.env_plugger_mot_ctrl import EnvironmentPluggerMotionCtrl
from gym_chargepal.envs.env_reacher_pos_ctrl import EnvironmentReacherPositionCtrl
from gym_chargepal.envs.env_plugger_pos_ctrl import EnvironmentPluggerPositionCtrl
from gym_chargepal.envs.env_plugger_cop_ctrl import EnvironmentPluggerComplianceCtrl

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
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.15, 0.3, -0.15)),
        # Reset variance of the start pose
        'variance': ((0.1, 0.1, 0.1), (0.0, 0.0, 0.0)), 
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
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.15, 0.3, -0.15)),
        # Reset variance of the start pose
        'variance': ((0.1, 0.1, 0.1), (0.1 * np.pi, 0.1 * np.pi, 0.1 * np.pi)), 
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
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (0.01 * np.pi, 0.01 * np.pi, 0.01 * np.pi)), 
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },

    'low_level_control': {
        'wa_lin': 0.05,          # action scaling in linear directions [m]
        'wa_ang': 0.05 * np.pi,  # action scaling in angular directions [rad]
    },
}


# ////////////////////////////////////////////////////////// #
# ///                                                    /// #
# ///   Environments Plugger with TCP force controller   /// #
# ///                                                    /// #
# ////////////////////////////////////////////////////////// #

testbed_plugger_6dof_force_ctrl_v1 = {
    'environment': {
        'type': EnvironmentPluggerForceCtrl,
        'T': 300,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (0.01 * np.pi, 0.01 * np.pi, 0.01 * np.pi)),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },
}


# /////////////////////////////////////////////////////////// #
# ///                                                     /// #
# ///   Environments Plugger with TCP motion controller   /// #
# ///                                                     /// #
# /////////////////////////////////////////////////////////// #

testbed_plugger_6dof_motion_ctrl_v1 = {
    'environment': {
        'type': EnvironmentPluggerMotionCtrl,
        'T': 200,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (0.01 * np.pi, 0.01 * np.pi, 0.01 * np.pi)),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },
}


# /////////////////////////////////////////////////////////////// #
# ///                                                         /// #
# ///   Environments Plugger with TCP compliance controller   /// #
# ///                                                         /// #
# /////////////////////////////////////////////////////////////// #

testbed_plugger_6dof_compliance_ctrl_v1 = {
    'environment': {
        'type': EnvironmentPluggerComplianceCtrl,
        'T': 500,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(12,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # 'X_tgt2plug': Pose().from_xyz((0.06, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (0.01 * np.pi, 0.01 * np.pi, 0.01 * np.pi)),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },
}
