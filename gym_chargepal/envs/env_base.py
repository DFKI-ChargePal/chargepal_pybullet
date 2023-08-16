""" This file defines the environments base class. """
from __future__ import annotations

# global
import abc
import numpy as np
# import gymnasium as gym
from gymnasium import Env, spaces
from rigmopy import Pose
from dataclasses import dataclass

# local
from gym_chargepal.utility.env_clock import EnvironmentClock
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any, Generic, TypeVar
from numpy import typing as npt

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")



@dataclass
class EnvironmentCfg(Generic[ObsType, ActType], ConfigHandler):
    T: int = 100  # time horizon
    task_pos_eps: float = 0.003  # position task criterion [m]
    task_ang_eps: float = 0.0175  # angular task criterion [rad]
    action_space: spaces.Space[ActType] | None = None
    observation_space: spaces.Space[ObsType] | None = None
    start_config: Pose = Pose()
    reset_variance: tuple[tuple[float, ...], tuple[float, ...]] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class Environment(Env[ObsType, ActType], Generic[ObsType, ActType], metaclass=abc.ABCMeta):
    """
    This is the concrete implementation of the OpenAI gym interface.
    """

    def __init__(self, config: dict[str, Any], render_mode: str | None = None):
        # Create configuration object
        self.cfg: EnvironmentCfg[ObsType, ActType] = EnvironmentCfg()
        self.cfg.update(**config)
        # Simulation parameter
        self.clock = EnvironmentClock(self.cfg.T)
        # Get action and observation space
        if self.cfg.action_space is not None:
            self.action_space = self.cfg.action_space
        else:
            raise ValueError(f"Please define the action space in the configuration.")
        if self.cfg.observation_space is not None:
            self.observation_space = self.cfg.observation_space
        else:
            raise ValueError(f"Please define the action space in the configuration.")
        # Render option can be enabled with the env.render() function
        self.is_render = False
        # self.toggle_render_mode = False
        # Performance logging
        self.task_pos_error = np.inf
        self.task_ang_error = np.inf
        if render_mode == 'human':
            self.render()

    @abc.abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[Any, Any]]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def close(self) -> None:
        """ Environment close function called at the end of the program. """
        raise NotImplementedError('Must be implemented in subclass.')

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            self.is_render = True
        else:
            self.is_render = False

    @property
    def done(self) -> bool:
        """ Check if environment is at the end of the episode. """
        return self.clock.check_for_time_horizon()

    @property
    def solved(self) -> bool:
        """ Check if spatial error is small enough """
        solved = False
        if self.task_pos_error < self.cfg.task_pos_eps and self.task_ang_error < self.cfg.task_ang_eps:
            solved = True
        return solved
