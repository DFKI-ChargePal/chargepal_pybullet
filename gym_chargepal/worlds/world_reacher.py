""" This file defines the reacher worlds. """
# global
import os
import logging
import numpy as np
from rigmopy import Orientation, Pose, Position

# local
from gym_chargepal.bullet.ur_arm import URArm
import gym_chargepal.utility.cfg_handler as ch
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.worlds.world import WorldCfg, World

# mypy
from typing import Any, Dict, Tuple, Union


LOGGER = logging.getLogger(__name__)


class WorldReacherCfg(WorldCfg):
    plane_urdf: str = 'plane.urdf'
    robot_urdf: str = 'primitive_chargepal_with_fix_plug.urdf'
    plane_config: Pose = Pose()
    robot_config: Pose = Pose(Position().from_xyz((0.0, 1.15, 0.0)))
    target_config: Pose = Pose(Position().from_xyz((0.0, 0.0, 1.2)), Orientation().from_euler_angle((np.pi/2, 0.0, 0.0)))


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
        ur_arm_config = ch.search(config, 'ur_arm')
        self.ur_arm = URArm(ur_arm_config)
        # Extract start configurations
        self.target_pose = self.cfg.target_config
        self.plane_pos, self.plane_ori = self.cfg.plane_config.xyz_xyzw
        self.robot_pos, self.robot_ori = self.cfg.robot_config.xyz_xyzw

    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None, render: bool = False) -> None:
        if self.bullet_client is None:
            # Connect to bullet simulation server
            self.connect(render)
            assert self.bullet_client
            # Load plane
            self.plane_id = self.bullet_client.loadURDF(
                fileName=self.cfg.plane_urdf,
                basePosition=self.plane_pos,
                baseOrientation=self.plane_ori
                )
            # Load robot
            f_path_robot_urdf = os.path.join(self.urdf_pkg_path, self.cfg.robot_urdf)
            self.robot_id = self.bullet_client.loadURDF(
                fileName=f_path_robot_urdf,
                basePosition=self.robot_pos,
                baseOrientation=self.robot_ori
                )
            # Set gravity
            self.bullet_client.setGravity(*self.cfg.gravity)
            # Create bullet body helper objects
            self.ur_arm.connect(self.bullet_client, self.robot_id)
        
        self.ur_arm.reset(joint_cfg=joint_conf)
        self.ur_arm.update()
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
