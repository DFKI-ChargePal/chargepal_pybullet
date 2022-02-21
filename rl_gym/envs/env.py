""" This file defines the environments base class. """
import abc
import copy
import gym
import numpy as np

# mypy
from typing import Dict, Any, Union, Tuple

from chargepal_pybullet.rl_gym.envs.config import ENVIRONMENT


class Environment(gym.Env):  # type: ignore
    """
    This is the concrete implementation of the OpenAI gym interface.
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(ENVIRONMENT)
        config.update(hyperparams)
        self._hyperparams = config
        super(Environment, self).__init__()
        # simulation parameter
        self._n_step = 0
        # define action and observation space
        self.action_space = self._hyperparams['action_space']
        self.observation_space = self._hyperparams['observation_space']
        # seed
        self._rs = None
        self.seed(123456789)

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def exit(self) -> None:
        """ Environment exit function called at the end of the program. """
        raise NotImplementedError('Must be implemented in subclass.')

    def seed(self, seed: Union[None, int, np.ndarray] = None) -> Union[None, int, np.ndarray]:
        self._rs = np.random.RandomState(seed)
        return seed

    def render(self, mode: str = "human") -> None:
        pass
