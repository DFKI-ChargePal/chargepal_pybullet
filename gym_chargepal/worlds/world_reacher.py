""" This file defines the reacher worlds. """
# global
import os
import logging
import numpy as np
from rigmopy import Pose, Quaternion, Vector3d

# local
from gym_chargepal.bullet.ur_arm import URArm
import gym_chargepal.utility.cfg_handler as ch
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.worlds.world import WorldCfg, World
from gym_chargepal.bullet.ref_body import ReferenceBody

# mypy
from typing import Any, Dict, Tuple, Union


LOGGER = logging.getLogger(__name__)


class WorldReacherCfg(WorldCfg):
    plane_urdf: str = 'plane.urdf'
    env_urdf: str = 'testbed_table_cic.urdf'
    robot_urdf: str = 'ur10e_fix_plug.urdf'
    plane_config: Pose = Pose()
    env_config: Pose = Pose().from_xyz((0.0, 1.15, 0.0))
    robot_config: Pose = Pose().from_xyz((0.0, 1.15, 0.0))
    target_config: Pose = Pose().from_xyz((0.0, 0.0, 1.2)).from_euler_angle(angles=(np.pi/2, 0.0, 0.0))


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
        ref_body_config = ch.search(config, 'ref_body')
        self.ref_body = ReferenceBody(ref_body_config)
        ur_arm_config = ch.search(config, 'ur_arm')
        self.ur_arm = URArm(ur_arm_config, self.ref_body)
        # Extract start configurations
        self.X_arm2tgt = self.cfg.target_config
        self.X_world2tgt = Pose()
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
            self.ref_body.connect(self.bullet_client, self.robot_id)
            self.ur_arm.connect(self.bullet_client, self.robot_id)
        
        self.ur_arm.reset(joint_cfg=joint_conf)
        self.ref_body.update()
        self.ur_arm.update()
        X_world2arm = self.ref_body.link.get_pose_ref()
        self.X_world2tgt = X_world2arm * self.X_arm2tgt
        self.draw_target(render)

    def sub_step(self) -> None:
        self.ur_arm.update()

    def draw_target(self, render: bool) -> None:
        if render:
            if self.target_id > -1:
                self.bullet_client.removeBody(self.target_id)
            self.target_id = pb_utils.draw_cylinder_marker(
                pose=self.X_world2tgt,
                radius=0.035, 
                height=0.080,
                color=(1, 0, 0, 0.75), 
                bullet_client=self.bullet_client
                )
