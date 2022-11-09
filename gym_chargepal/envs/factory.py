""" This file contains the environment factory. The entry point for building gym environments"""
# global
import copy
import logging

# local
from gym_chargepal.envs.env_base import Environment

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


class EnvironmentFactory:

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __new__(cls: type, **kwargs: Any) -> Environment:
        """
            create a new gym environment
        """
        # remap
        config = copy.deepcopy(kwargs)
        # extract environment type
        env_type = config['environment']['type']
        # create environment
        env: Environment = env_type(**config)
        return env
