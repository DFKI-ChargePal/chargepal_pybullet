""" This file contains a register of all Environments with its specific configuration. """
import copy
import numpy as np
from gym import spaces


# mypy
from typing import Any, Dict


# base environments
from gym_chargepal.envs.env_ptp_pos_ctrl import EnvironmentTcpPositionCtrlPtP
from gym_chargepal.envs.env_ptp_vel_ctrl import EnvironmentTcpVelocityCtrlPtP


def update_kwargs_dict(kwargs_dict: Dict[str, Any], config_name: str, config_dict: Dict[str, Any]) -> None:
        """ helper function to forward the default configuration arguments """
        config_cpy = copy.deepcopy(config_dict)
        if config_name in kwargs_dict:
            config_cpy.update(kwargs_dict[config_name])
        kwargs_dict[config_name] = config_cpy


""" Concrete point-to-point position controlled environments. """
# Constants
DEFAULT_PLUG_ANG_POS = (np.pi/2, np.pi/2, 0.0)
DEFAULT_TARGET_LIN_POS = (0.0, 0.0, 1.2)


class EnvironmentTcpPositionCtrlPtP1Dof(EnvironmentTcpPositionCtrlPtP):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(1,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'tgt_config_pos': DEFAULT_TARGET_LIN_POS,
        'tgt_config_ang': DEFAULT_PLUG_ANG_POS,
        # Start configuration relative to target configuration
        'start_config_pos': (0.0, 0.2, 0.0),
        'start_config_ang': (0.0, 0.0, 0.0),
        # Reset variance of the start pose
        'reset_variance': ((0.0, 0.05, 0.0), (0.0, 0.0, 0.0)),
        # Task performance criteria
        'task_epsilon_pos': 0.003,  # 3 mm
        'task_epsilon_ang': 1,  # 1 rad
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


class EnvironmentTcpPositionCtrlPtP3Dof(EnvironmentTcpPositionCtrlPtP):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'tgt_config_pos': DEFAULT_TARGET_LIN_POS,
        'tgt_config_ang': DEFAULT_PLUG_ANG_POS,
        # Start configuration relative to target configuration
        'start_config_pos': (0.0, 0.2, 0.0),
        'start_config_ang': (0.0, 0.0, 0.0),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.05, 0.01), (0.0, 0.0, 0.0)),
        # Task performance criteria
        'task_epsilon_pos': 0.003,  # 3 mm
        'task_epsilon_ang': 1,  # 1 rad
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


class EnvironmentTcpPositionCtrlPtP6Dof(EnvironmentTcpPositionCtrlPtP):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(7,), dtype=np.float32),
        # Target configuration
        'tgt_config_pos': DEFAULT_TARGET_LIN_POS,
        'tgt_config_ang': DEFAULT_PLUG_ANG_POS,
        # Start configuration relative to target configuration
        'start_config_pos': (0.0, 0.2, 0.0),
        'start_config_ang': (0.0, 0.0, 0.0),
        # Reset variance of the start pose
        'reset_variance': ((0.05, 0.05, 0.05), (0.1, 0.1, 0.1)),
        # Task performance criteria
        'task_epsilon_pos': 0.003,  # 3 mm
        'task_epsilon_ang': 1,  # 1 rad
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


class EnvironmentTcpVelocityCtrlPtP1Dof(EnvironmentTcpVelocityCtrlPtP):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(1,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
        # Target configuration
        'tgt_config_pos': DEFAULT_TARGET_LIN_POS,
        'tgt_config_ang': DEFAULT_PLUG_ANG_POS,
        # Start configuration relative to target configuration
        'start_config_pos': (0.0, 0.2, 0.0),
        'start_config_ang': (0.0, 0.0, 0.0),
        # Reset variance of the start pose
        'reset_variance': ((0.0, 0.05, 0.0), (0.0, 0.0, 0.0)),
        # Task performance criteria
        'task_epsilon_pos': 0.003,  # 3 mm
        'task_epsilon_ang': 1,  # 1 rad
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'wa_lin': 0.125,  # action scaling in linear directions [m]
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


class EnvironmentTcpVelocityCtrlPtP3Dof(EnvironmentTcpVelocityCtrlPtP):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(3,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
        # Target configuration
        'tgt_config_pos': DEFAULT_TARGET_LIN_POS,
        'tgt_config_ang': DEFAULT_PLUG_ANG_POS,
        # Start configuration relative to target configuration
        'start_config_pos': (0.0, 0.2, 0.0),
        'start_config_ang': (0.0, 0.0, 0.0),
        # Reset variance of the start pose
        'reset_variance': ((0.01, 0.05, 0.01), (0.0, 0.0, 0.0)),
        # Task performance criteria
        'task_epsilon_pos': 0.003,  # 3 mm
        'task_epsilon_ang': 1,  # 1 rad
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'wa_lin': 0.05,  # action scaling in linear directions [m]
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


class EnvironmentTcpVelocityCtrlPtP6Dof(EnvironmentTcpVelocityCtrlPtP):

    # configuration environment
    config_env = {
        'T': 100,
        'action_space': spaces.Box(low=-1.0,  high=1.0, shape=(6,), dtype=np.float32),
        'observation_space': spaces.Box(low=-1.0,  high=1.0, shape=(13,), dtype=np.float32),
        # Target configuration
        'tgt_config_pos': DEFAULT_TARGET_LIN_POS,
        'tgt_config_ang': DEFAULT_PLUG_ANG_POS,
        # Start configuration relative to target configuration
        'start_config_pos': (0.0, 0.2, 0.0),
        'start_config_ang': (0.0, 0.0, 0.0),
        # Reset variance of the start pose
        'reset_variance': ((0.05, 0.05, 0.05), (0.1, 0.1, 0.1)),
        # Task performance criteria
        'task_epsilon_pos': 0.003,  # 3 mm
        'task_epsilon_ang': 1,  # 1 rad
        }

    config_world = {
        'hz_ctrl': 20,
    }

    # configuration low-level controller
    config_low_level_control = {
        'wa_lin': 0.05,  # action scaling in linear directions [m]
        'wa_ang': 0.05 * np.pi,  # action scaling in angular directions [rad]
        }

    def __init__(self, **kwargs: Any):
        # Update configuration
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_env', config_dict=self.config_env)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_world', config_dict=self.config_world)
        update_kwargs_dict(kwargs_dict=kwargs, config_name='config_low_level_control', config_dict=self.config_low_level_control)
        # Create environment
        super().__init__(**kwargs)
