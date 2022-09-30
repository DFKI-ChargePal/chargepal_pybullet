""" This file defines the top down experiment task in a cartesian controlled world. """
# global
import os
import copy 
import logging
import pybullet as p

# local
from gym_chargepal.worlds.world import World
from gym_chargepal.worlds.config import WORLD_TDT
import gym_chargepal.bullet.utility as bullet_helper


# mypy
from typing import Any, Dict, Union, Tuple


LOGGER = logging.getLogger(__name__)


class WorldTopDownTask(World):

    """ Build a testbed with a top down peg-in-hole task. """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD_TDT)
        config.update(hyperparams)
        super().__init__(config)

        # pybullet model ids
        self.plane_id = -1
        self.robot_id = -1
        self.socket_id = -1

        # links and joints
        self.ur_joint_idx_dict: Dict[str, int] = {}