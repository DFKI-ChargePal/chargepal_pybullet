""" This file defines the environments base class. """
import abc
import copy
import gym
import numpy as np
from numpy.random import RandomState

from gym_chargepal.utility.env_clock import EnvironmentClock

# mypy
from numpy import typing as npt
from typing import Dict, Any, Optional, Union, Tuple


class Environment(gym.Env):  # type: ignore
    """
    This is the concrete implementation of the OpenAI gym interface.
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        self.hyperparams = copy.deepcopy(hyperparams)
        super(Environment, self).__init__()
        # simulation parameter
        self.clock = EnvironmentClock(self.hyperparams['T'])
        # define action and observation space
        self.action_space = self.hyperparams['action_space']
        self.observation_space = self.hyperparams['observation_space']
        # seed
        self.rs: Optional[RandomState] = None
        self.seed()

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

    def seed(self, seed: Union[None, int, npt.NDArray[np.float32]] = None) -> Union[None, int, npt.NDArray[np.float32]]:
        self.rs = RandomState(seed)
        return seed

    def render(self, mode: str = "human") -> None:
        pass
