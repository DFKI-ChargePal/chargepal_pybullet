""" This file defines the environments base class. """
# global
import abc
import gym
import numpy as np
from dataclasses import dataclass

# local
from gym_chargepal.utility.tf import Quaternion, Translation, Pose, RndPoseGenerator
from gym_chargepal.utility.env_clock import EnvironmentClock
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from numpy import typing as npt
from typing import Dict, Any, Optional, Tuple

@dataclass
class EnvironmentCfg(ConfigHandler):
    T: int = 100  # time horizon
    task_pos_eps: float = 0.003  # position task criterion [m]
    task_ang_eps: float = 1.0  # angular task criterion [rad]
    action_space: Optional[gym.spaces.Space] = None
    observation_space: Optional[gym.spaces.Space] = None
    start_config: Pose = Pose(Translation(), Quaternion())
    target_config: Pose = Pose(Translation(), Quaternion()) 
    reset_variance: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class Environment(gym.Env, metaclass=abc.ABCMeta):
    """
    This is the concrete implementation of the OpenAI gym interface.
    """
    def __init__(self, config: Dict[str, Any]):
        # Call base class
        super(Environment, self).__init__()
        # Create configuration object
        self.cfg = EnvironmentCfg()
        self.cfg.update(**config)
        # Simulation parameter
        self.clock = EnvironmentClock(self.cfg.T)
        self.reset_rnd_gen = RndPoseGenerator(*self.cfg.reset_variance)
        # Get action and observation space
        self.action_space = self.cfg.action_space
        self.observation_space = self.cfg.observation_space
        # performance logging
        self.task_pos_error = np.inf
        self.task_ang_error = np.inf

    @abc.abstractmethod
    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def reset(self) -> npt.NDArray[np.float32]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def close(self) -> None:
        """ Environment close function called at the end of the program. """
        raise NotImplementedError('Must be implemented in subclass.')

    def render(self, mode: str = "human") -> None:
        pass

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
