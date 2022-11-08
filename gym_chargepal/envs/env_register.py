""" This file contains a register of all Environments with its specific configuration. """
import copy
import numpy as np
from gym import spaces


# local
from gym_chargepal.utility.tf import Quaternion, Translation, Pose
# base environments
from gym_chargepal.envs.env_reacher_pos_ctrl import EnvironmentReacherPositionCtrl
from gym_chargepal.envs.env_plugger_pos_ctrl import EnvironmentPluggerPositionCtrl
from gym_chargepal.envs.env_reacher_vel_ctrl import EnvironmentReacherVelocityCtrl

# mypy
from typing import Any, Dict



def update_kwargs_dict(kwargs_dict: Dict[str, Any], config_name: str, config_dict: Dict[str, Any]) -> None:
        """ helper function to forward the default configuration arguments """
        config_cpy = copy.deepcopy(config_dict)
        if config_name in kwargs_dict:
            config_cpy.update(kwargs_dict[config_name])
        kwargs_dict[config_name] = config_cpy


""" Concrete point-to-point position controlled environments. """
# Constants
ROT_PI_2_X = Quaternion()
ROT_PI_2_X.from_euler_angles(np.pi/2, 0.0, 0.0)

ROT_PI_X = Quaternion()
ROT_PI_X.from_euler_angles(np.pi, 0.0, 0.0)

DEFAULT_TARGET_POS = Translation(0.0, 0.0, 1.2)


# ///////////////////////////////////////////////////// #
# ///                                               /// #
# ///   Environments with TCP position controller   /// #
# ///                                               /// #
# ///////////////////////////////////////////////////// #
class EnvironmentReacherPositionCtrl1Dof(EnvironmentReacherPositionCtrl):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(1,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.2, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.0, 0.025, 0.0), (0.0, 0.0, 0.0)),
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'wa_lin': 0.025,  # action scaling in linear directions [m]
        'linear_enabled_motion_axis': (False, True, False),
        'angular_enabled_motion_axis': (False, False, False),
        }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_low_level_control', config_dict=self.config_low_level_control)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentReacherPositionCtrl3Dof(EnvironmentReacherPositionCtrl):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.2, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.05, 0.01), (0.0, 0.0, 0.0)),
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'wa_lin': 0.025,  # action scaling in linear directions [m]
        'linear_enabled_motion_axis': (True, True, True),
        'angular_enabled_motion_axis': (False, False, False),
        }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_low_level_control', config_dict=self.config_low_level_control)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentReacherPositionCtrl6Dof(EnvironmentReacherPositionCtrl):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.2, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.05, 0.05, 0.05), (0.1, 0.1, 0.1)),
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'wa_lin': 0.025,  # action scaling in linear directions [m]
        'wa_ang': 0.05,  # action scaling in angular directions [rad]
        }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_low_level_control', config_dict=self.config_low_level_control)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentPluggerPositionCtrl6DofV0(EnvironmentPluggerPositionCtrl):

    # configuration environment
    config_env = {
        'T': 200,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.1, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.005, 0.01), (0.05, 0.05, 0.05)),
        }

    config_world = {
        'hz_ctrl': 40,
    }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        # Create environment
        super().__init__(**kwargs)


SOCKET_POS = Translation(0.0, -0.1/2.0, 0.0)
SOCKET_ORI = Quaternion()
SOCKET_ORI.from_euler_angles(0.0, 0.0, np.pi)
class EnvironmentPluggerPositionCtrl6DofV1(EnvironmentPluggerPositionCtrl):

    # configuration environment
    config_env = {
        'T': 200,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.075, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.005, 0.01), (0.05, 0.05, 0.05)),
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        }

    rod_diameter = "32d5"
    config_world = {
        'hz_ctrl': 40,
        'robot_urdf': f'cpm_fix_rod_{rod_diameter}.urdf',
        'socket_urdf': 'adapter_station_square_socket.urdf',
        'socket_config': Pose(SOCKET_POS, SOCKET_ORI),
    }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentPluggerPositionCtrl6DofV2(EnvironmentPluggerPositionCtrl):

    # configuration environment
    config_env = {
        'T': 200,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.075, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.005, 0.01), (0.05, 0.05, 0.05)),
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        }
    rod_diameter = "34"
    config_world = {
        'hz_ctrl': 40,
        'robot_urdf': f'cpm_fix_rod_{rod_diameter}.urdf',
        'socket_urdf': 'adapter_station_square_socket.urdf',
        'socket_config': Pose(SOCKET_POS, SOCKET_ORI),
    }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentTestbedPluggerPositionCtrl6DofV0(EnvironmentPluggerPositionCtrl):

    # configuration environment
    config_env = {
        'T': 200,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(Translation(0.5, 0.8, 0.0), ROT_PI_X),
        # Start configuration relative to target configuration
        'start_config': Pose(Translation(0.0, 0.0, 0.02+0.095), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.01, 0.001), (0.05, 0.05, 0.05)),
        # 'reset_variance': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        }

    config_world = {
        'hz_ctrl': 40,
        'robot_urdf': 'chargepal_testbed_tdt.urdf',
        'socket_urdf': 'tdt_socket.urdf',
        'plane_config': Pose(Translation(0.0, 0.0, -0.8136), Quaternion()),
        'robot_config': Pose(Translation(), Quaternion()),
        'socket_config': Pose(Translation(0.5, 0.8, 0.0), Quaternion()),
    }

    config_ur_arm = {
        'joint_default_values': {
            'shoulder_pan_joint': np.pi,
            'shoulder_lift_joint': -7*np.pi/36,
            'elbow_joint': -np.pi/2 - 7*np.pi/36,
            'wrist_1_joint': -np.pi/2 - 4*np.pi/36,
            'wrist_2_joint': np.pi/2,
            'wrist_3_joint': -np.pi/2,
        },
    }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_ur_arm', config_dict=self.config_ur_arm)
        # Create environment
        super().__init__(**kwargs)


# ///////////////////////////////////////////////////// #
# ///                                               /// #
# ///   Environments with TCP velocity controller   /// #
# ///                                               /// #
# ///////////////////////////////////////////////////// #
class EnvironmentReacherVelocityCtrl1Dof(EnvironmentReacherVelocityCtrl):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(1,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.2, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.0, 0.05, 0.0), (0.0, 0.0, 0.0)),
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'linear_enabled_motion_axis': (False, True, False),
        'angular_enabled_motion_axis': (False, False, False),
        }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_low_level_control', config_dict=self.config_low_level_control)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentReacherVelocityCtrl3Dof(EnvironmentReacherVelocityCtrl):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.2, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.05, 0.01), (0.0, 0.0, 0.0)),
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'linear_enabled_motion_axis': (True, True, True),
        'angular_enabled_motion_axis': (False, False, False),
        }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_low_level_control', config_dict=self.config_low_level_control)
        # Create environment
        super().__init__(**kwargs)


class EnvironmentReacherVelocityCtrl6Dof(EnvironmentReacherVelocityCtrl):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
        # Target configuration
        'target_config': Pose(DEFAULT_TARGET_POS, ROT_PI_2_X),
        # Start configuration relative to target configuration'
        'start_config': Pose(Translation(0.0, 0.2, 0.0), Quaternion()),
        # Reset variance of the start pose
        'reset_variance': ((0.05, 0.05, 0.05), (0.1, 0.1, 0.1)),
        }

    config_world = {
        'hz_ctrl': 20,
    }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        # Create environment
        super().__init__(**kwargs)
