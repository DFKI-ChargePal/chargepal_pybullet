""" This file contains the environment factory. The entry point for building gym environments"""
# global
import copy
import logging

# local
from gym_chargepal.envs.env_base import Environment, ObsType, ActType

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


class EnvironmentFactory:

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __new__(cls: type, **kwargs: Any) -> Environment[ObsType, ActType]:  # type: ignore
        """
            create a new gym environment
        """
        # remap
        config = copy.deepcopy(kwargs)
        if 'kwargs' in config:
            config_kwargs = copy.deepcopy(config['kwargs'])
            del config['kwargs']

            for class_cfg in config:
                if class_cfg in config_kwargs:
                    config[class_cfg].update(config_kwargs[class_cfg])

        # extract environment type
        env_type = config['environment']['type']
        # create environment
        env: Environment[ObsType, ActType] = env_type(**config)
        return env
