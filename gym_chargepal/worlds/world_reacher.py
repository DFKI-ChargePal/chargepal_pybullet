""" This file defines the reacher worlds. """
from __future__ import annotations

# global
import logging
from rigmopy import Pose

# local
from gym_chargepal.bullet.ur_arm import URArm
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.worlds.world import WorldCfg, World
from gym_chargepal.utility.start_sampler import StartSampler
from gym_chargepal.utility.virtual_target import VirtualTarget

# mypy
from typing import Any

LOGGER = logging.getLogger(__name__)

_TABLE_HEIGHT = 0.8136


class WorldReacherCfg(WorldCfg):
    plane_urdf: str = 'plane.urdf'
    env_urdf: str = 'testbed_table_cic.urdf'
    robot_urdf: str = 'ur10e_fix_plug.urdf'
    plane_config: Pose = Pose().from_xyz((0.0, 0.0, -_TABLE_HEIGHT))
    env_config: Pose = Pose()

class WorldReacher(World):

    def __init__(self,
                 config:       dict[str, Any], 
                 config_arm:   dict[str, Any], 
                 config_start: dict[str, Any], 
                 config_tgt:   dict[str, Any]
                 ) -> None:
        """ Build a robot world where the task is to reach a point from a random start configuration

        Args:
            config:       Dictionary to overwrite values of the world configuration class
            config_arm:   Dictionary to overwrite values of the UR arm configuration class
            config_start: Dictionary to overwrite values of the start sampler configuration class
            config_tgt:   Dictionary to overwrite values of the virtual target configuration class
        """
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: WorldReacherCfg = WorldReacherCfg()
        self.cfg.update(**config)
        # Pre initialize class attributes
        self.env_id = -1
        self.plane_id = -1
        self.robot_id = -1
        self.target_id = -1
        self.ur_arm = URArm(config_arm)
        self.start = StartSampler(config_start)
        self.vrt_tgt = VirtualTarget(config_tgt)

    def sample_X0(self) -> Pose:
        X0_world2plug = self.ur_arm.X_world2arm * self.vrt_tgt.X_arm2tgt * self.start.random_X_tgt2plug
        return X0_world2plug

    def reset(self, joint_conf: tuple[float, ...] | None = None, render: bool = False) -> None:
        if self.bullet_client is None:
            # Connect to bullet simulation server
            self.connect(render)
            assert self.bullet_client
            # Load plane
            self.plane_id = self.bullet_client.loadURDF(
                fileName=self.cfg.plane_urdf,
                basePosition=self.cfg.plane_config.xyz,
                baseOrientation=self.cfg.plane_config.xyzw
                )
            # Load environment
            self.env_id = self.bullet_client.loadURDF(
                fileName=str(self.urdf_pkg_path.joinpath(self.cfg.env_urdf)),
                basePosition=self.cfg.env_config.xyz,
                baseOrientation=self.cfg.env_config.xyzw
            )
            # Load robot
            self.robot_id = self.bullet_client.loadURDF(
                fileName=str(self.urdf_pkg_path.joinpath(self.cfg.robot_urdf)),
                basePosition=self.ur_arm.X_world2arm.xyz,
                baseOrientation=self.ur_arm.X_world2arm.xyzw
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
                pose=self.ur_arm.base_link.X_world2link * self.vrt_tgt.X_arm2tgt,
                radius=0.035, 
                height=0.080,
                color=(1, 0, 0, 0.75), 
                bullet_client=self.bullet_client
                )
