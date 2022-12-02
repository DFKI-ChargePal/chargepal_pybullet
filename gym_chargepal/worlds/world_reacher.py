""" This file defines the reacher worlds. """
# global
import os
import logging
import numpy as np

# local
import gym_chargepal.bullet.utility as pb_utils
import gym_chargepal.utility.cfg_handler as cfg_helper

from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.worlds.world import WorldCfg, World
from gym_chargepal.utility.tf import Quaternion, Translation, Pose

# mypy
from typing import Any, Dict, Tuple, Union


# constants
LOGGER = logging.getLogger(__name__)

TARGET_POS = Translation(0.0, 0.0, 1.2)
TARGET_ORI = Quaternion()
TARGET_ORI.from_euler_angles(np.pi/2, 0.0, 0.0)

ROBOT_ORI = Quaternion()
ROBOT_POS = Translation(x=0.0, y=1.15, z=0.0)


class WorldReacherCfg(WorldCfg):
    robot_urdf: str = 'primitive_chargepal_with_fix_plug.urdf'
    robot_config: Pose = Pose(ROBOT_POS, ROBOT_ORI)
    target_config: Pose = Pose(TARGET_POS, TARGET_ORI)


class WorldReacher(World):

    """ Build a robot world where the task is to reach a point from a random start configuration """
    def __init__(self, config: Dict[str, Any]):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: WorldReacherCfg = WorldReacherCfg()
        self.cfg.update(**config)

        # Pre initialize class attributes
        self.plane_id = -1
        self.robot_id = -1
        self.target_id = -1
        ur_arm_config = cfg_helper.search(config, 'ur_arm')
        self.ur_arm = URArm(ur_arm_config)

        # Extract start configurations
        self.robot_pos = self.cfg.robot_config.pos.as_tuple()
        self.robot_ori = self.cfg.robot_config.ori.as_tuple(order='xyzw')
        self.target_pose = self.cfg.target_config

    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None, render: bool = False) -> None:
        if self.bullet_client is None:
            # connect to bullet simulation server
            self.connect(render)
            assert self.bullet_client
            # load plane
            self.plane_id = self.bullet_client.loadURDF('plane.urdf')
            # load robot
            f_path_robot_urdf = os.path.join(self.urdf_pkg_path, self.cfg.robot_urdf)
            self.robot_id = self.bullet_client.loadURDF(
                fileName=f_path_robot_urdf,
                basePosition=self.robot_pos,
                baseOrientation=self.robot_ori
                )
            # set gravity
            self.bullet_client.setGravity(*self.cfg.gravity)
            # create bullet body helper objects
            self.ur_arm.connect(self.bullet_client, self.robot_id)

        self.ur_arm.reset(joint_cfg=joint_conf)
        self.ur_arm.update()
        # draw target
        self.draw_target(render)

    def sub_step(self) -> None:
        self.ur_arm.update()

    def draw_target(self, render: bool) -> None:
        if render:
            if self.target_id > -1:
                self.bullet_client.removeBody(self.target_id)
            self.target_id = pb_utils.draw_cylinder_marker(
                pose=self.target_pose,
                radius=0.035, 
                height=0.080,
                color=(1, 0, 0, 0.75), 
                bullet_client=self.bullet_client
                )
