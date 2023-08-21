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
from gym_chargepal.worlds.world import World
from gym_chargepal.bullet.ik_solver import IKSolver
from gym_chargepal.controllers.controller import Controller
from gym_chargepal.utility.cfg_handler import ConfigHandler
from gym_chargepal.utility.env_clock import EnvironmentClock

# typing
from typing import Any, Generic, TypeVar
from numpy import typing as npt

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


@dataclass
class EnvironmentCfg(Generic[ObsType, ActType], ConfigHandler):
    T: int = 100  # time horizon
    render_mode: str | None = None
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

    def __init__(self, config: dict[str, Any]):
        # Create configuration object
        self.cfg: EnvironmentCfg[ObsType, ActType] = EnvironmentCfg()
        self.cfg.update(**config)
        # Simulation parameter
        self.clock = EnvironmentClock(self.cfg.T)
        # Core object each child class must have
        self._world: World | None = None
        self._ik_solver: IKSolver | None = None
        self._controller: Controller | None = None
        # Get action and observation space
        if self.cfg.action_space is not None:
            self.action_space = self.cfg.action_space
        else:
            raise ValueError(f"Please define the action space in the configuration.")
        if self.cfg.observation_space is not None:
            self.observation_space = self.cfg.observation_space
        else:
            raise ValueError(f"Please define the action space in the configuration.")
        self.truncated = False
        # Render option can be enabled with the env.render() function
        self.is_render = False
        # self.toggle_render_mode = False
        # Performance logging
        self.task_pos_error = np.inf
        self.task_ang_error = np.inf
        if self.cfg.render_mode == 'human':
            self.render()

    @property
    def world(self) -> World:
        if self._world is None:
            raise ValueError(f"Please initialize the environment world before using it.")
        return self._world

    @world.setter
    def world(self, world: World) -> None:
        self._world = world

    @property
    def ik_solver(self) -> IKSolver:
        if self._ik_solver is None:
            raise ValueError(f"Please initialize the IK solver before using it.")
        return self._ik_solver

    @ik_solver.setter
    def ik_solver(self, solver: IKSolver) -> None:
        self._ik_solver = solver

    @property
    def controller(self) -> Controller:
        if self._controller is None:
            raise ValueError(f"Please initialize the controller before using it.")
        return self._controller
    
    @controller.setter
    def controller(self, ctrl: Controller) -> None:
        self._controller = ctrl

    @property
    def terminated(self) -> bool:
        """ Check if environment is at the end of the episode. """
        return self.clock.check_for_time_horizon()

    @property
    def solved(self) -> bool:
        """ Check if spatial error is small enough """
        solved = False
        if self.task_pos_error < self.cfg.task_pos_eps and self.task_ang_error < self.cfg.task_ang_eps:
            solved = True
        return solved

    def _reset_core(self) -> tuple[npt.NDArray[np.float32], dict[Any, Any]]:
        # Reset environment
        self.clock.reset()
        # Reset robot by default joint configuration
        self.world.reset(render=self.is_render)
        # Get start joint start configuration by inverse kinematic
        joint_pos0 = self.ik_solver.solve(self.world.sample_X0())
        self.world.reset(joint_conf=joint_pos0, render=self.is_render)
        self.controller.reset()
        return self.get_obs(), self.compose_info()

    def _update_core(self, 
                     action: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], bool, bool, dict[Any, Any]]:
        # Apply action
        self.controller.update(action=np.array(action))
        # Step simulation and clock
        self.world.step(render=self.is_render)
        self.clock.tick()
        return self.get_obs(), self.terminated, self.truncated, self.compose_info()

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            self.is_render = True
        else:
            self.is_render = False

    def close(self) -> None:
        self.world.disconnect()

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[Any, Any]]:
        """ Environment step function of the Reinforcement Learning framework. """
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def get_obs(self) -> npt.NDArray[np.float32]:
        """ Environment helper function to gather the observation """
        raise NotImplementedError('Must be implemented in subclass')

    @abc.abstractmethod
    def compose_info(self) -> dict[str, Any]:
        """ Environment helper function to gather the information signal """
        raise NotImplementedError('Must be implemented in subclass')
