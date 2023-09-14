""" Configuration setups point-to-point position controlled environments. """
# global
import numpy as np
from gymnasium import spaces
from rigmopy import Pose

# local
from gym_chargepal.envs.env_plugger_frc_ctrl import EnvironmentPluggerForceCtrl
from gym_chargepal.envs.env_plugger_mot_ctrl import EnvironmentPluggerMotionCtrl
from gym_chargepal.envs.env_reacher_pos_ctrl import EnvironmentReacherPositionCtrl
from gym_chargepal.envs.env_plugger_pos_ctrl import EnvironmentPluggerPositionCtrl
from gym_chargepal.envs.env_plugger_cop_ctrl import EnvironmentPluggerComplianceCtrl
from gym_chargepal.envs.env_searcher_cop_ctrl import EnvironmentSearcherComplianceCtrl
from gym_chargepal.envs.env_guided_searcher_cop_ctrl import EnvironmentGuidedSearcherComplianceCtrl

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
        'X_tgt2plug': Pose().from_xyz((0.1, 0.2, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.05, 0.05, 0.05), (0.0, 0.0, 0.0)), 
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
        'X_tgt2plug': Pose().from_xyz((0.1, 0.2, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.05, 0.05, 0.05), (np.deg2rad(5.0), np.deg2rad(5.0), np.deg2rad(5.0))), 
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
        'T': 200,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (np.deg2rad(2.5), np.deg2rad(2.5), np.deg2rad(2.5))), 
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
        'T': 250,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (np.deg2rad(2.5), np.deg2rad(2.5), np.deg2rad(2.5))), 
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
        'T': 250,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (np.deg2rad(2.5), np.deg2rad(2.5), np.deg2rad(2.5))), 
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
        'T': 250,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(12,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.1)),
        # Reset variance of the start pose
        'variance': ((0.025, 0.025, 0.01), (np.deg2rad(2.5), np.deg2rad(2.5), np.deg2rad(2.5))), 
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },
}


# //////////////////////////////////////////////////////////////// #
# ///                                                          /// #
# ///   Environments Searcher with TCP compliance controller   /// #
# ///                                                          /// #
# //////////////////////////////////////////////////////////////// #

testbed_searcher_6dof_compliance_ctrl_v0 = {
    'environment': {
        'type': EnvironmentSearcherComplianceCtrl,
        'T': 180,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(12,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.03)),
        # Reset variance of the start pose
        'variance': ((0.015, 0.015, 0.0), (np.deg2rad(2.5), np.deg2rad(2.5), np.deg2rad(2.5))),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },
}


# /////////////////////////////////////////////////////////////////////// #
# ///                                                                 /// #
# ///   Environments Guided Searcher with TCP compliance controller   /// #
# ///                                                                 /// #
# /////////////////////////////////////////////////////////////////////// #

testbed_guided_searcher_6dof_compliance_ctrl_v0 = {
    'environment': {
        'type': EnvironmentGuidedSearcherComplianceCtrl,
        'T': 180,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
    },

    'low_level_control': {
        'force_action_scale_lin': 0.5,
        'force_action_scale_ang': 1.0,
        'motion_action_scale_lin': 0.0125,
        'motion_action_scale_ang': 0.0125,
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.05)),
        # Reset variance of the start pose
        'variance': ((0.015, 0.015, 0.0), (np.deg2rad(2.0), np.deg2rad(2.0), np.deg2rad(2.0))),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.50, 0.50)).from_euler_angle((0.0, np.pi/2, 0.0)),
    },

    'socket_sensor': {
        'var_ang': ((2/180) * np.pi, (2/180) * np.pi, (2/180) * np.pi),
    },

}

testbed_guided_searcher_6dof_compliance_ctrl_v1 = {
    'environment': {
        'type': EnvironmentGuidedSearcherComplianceCtrl,
        'T': 180,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
    },

    'world': {
        'robot_urdf': 'ur10e_ccs_type2.urdf',
        'socket_urdf': 'ccs_socket_wall.urdf',
    },

    'ur_arm': {
        'tcp_link_name': 'plug_tip',
        'tool_com_link_names': ('plug_mounting',),
    },

    'low_level_control': {
        'force_action_scale_lin': 0.5,
        'force_action_scale_ang': 1.0,
        'motion_action_scale_lin': 0.0125,
        'motion_action_scale_ang': 0.0125,
    },

    'start': {
        # Start configuration w.r.t. target configuration'
        'X_tgt2plug': Pose().from_xyz((0.0, 0.0, -0.025)),
        # Reset variance of the start pose
        'variance': ((0.01, 0.01, 0.0), (np.deg2rad(1.0), np.deg2rad(1.0), np.deg2rad(1.0))),
    },

    'socket': {
        # Socket configuration w.r.t. the arm base
        'X_arm2socket': Pose().from_xyz((0.90, 0.30, 0.025)).from_euler_angle((np.pi, 0.0, 0.0)),
    },

    'socket_sensor': {
        'var_ang': ((2/180) * np.pi, (2/180) * np.pi, (2/180) * np.pi),
    },

}