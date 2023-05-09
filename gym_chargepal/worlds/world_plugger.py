""" This file defines the plugger worlds. """
from __future__ import annotations

# global
import os 
import logging
import numpy as np
from rigmopy import Pose

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.socket import Socket
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.worlds.world import WorldCfg, World

# mypy
from typing import Any


LOGGER = logging.getLogger(__name__)


_TABLE_HEIGHT = 0.8136
_TABLE_WIDTH = 0.81
_PROFILE_SIZE = 0.045
_BASE_PLATE_SIZE = 0.225
_BASE_PLATE_HEIGHT = 0.0225


class WorldPluggerCfg(WorldCfg):
    plane_urdf: str = 'plane.urdf'
    env_urdf: str = 'testbed_table_cic.urdf'
    robot_urdf: str = 'ur10e_fix_plug.urdf'
    socket_urdf: str = 'tdt_socket.urdf'
    plane_config: Pose = Pose().from_xyz((0.0, 0.0, -_TABLE_HEIGHT))
    env_config: Pose = Pose()
    robot_config: Pose = Pose().from_xyz(
        (_TABLE_WIDTH - _BASE_PLATE_SIZE/2, _PROFILE_SIZE + _BASE_PLATE_SIZE/2, _BASE_PLATE_HEIGHT)
        ).from_euler_angle(angles=(0.0, 0.0, np.pi/2))
    socket_config: Pose = Pose().from_xyz(
        (0.635 + 0.05, 0.319, 0.271)).from_euler_angle((0.0, -np.pi/2, 0.0))

class WorldPlugger(World):

    def __init__(self, config: dict[str, Any], config_arm: dict[str, Any], config_socket: dict[str, Any]):
        """ Build a robot world where the task is to connect some type of plug with some type of socket 

        Args:
            config: Dictionary to overwrite values of the world configuration class
        """
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: WorldPluggerCfg = WorldPluggerCfg()
        self.cfg.update(**config)
        # Pre initialize class attributes
        self.env_id = -1
        self.plane_id = -1
        self.robot_id = -1
        self.socket_id = -1
        config_arm['ft_buffer_size'] = self.sim_steps + 1
        self.ur_arm = URArm(config_arm)
        self.socket = Socket(config_socket)
        # Extract start configurations
        self.env_pos, self.env_ori = self.cfg.env_config.xyz_xyzw
        self.plane_pos, self.plane_ori = self.cfg.plane_config.xyz_xyzw
        self.robot_pos, self.robot_ori = self.cfg.robot_config.xyz_xyzw
        self.socket_pos, self.socket_ori = (self.cfg.robot_config * self.cfg.socket_config).xyz_xyzw

    def reset(self, joint_conf: tuple[float, ...] | None = None, render: bool = False) -> None:
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
            # Load environment
            f_path_env_urdf = os.path.join(self.urdf_pkg_path, self.cfg.env_urdf)
            self.env_id = self.bullet_client.loadURDF(
                fileName=f_path_env_urdf,
                basePosition=self.env_pos,
                baseOrientation=self.env_ori
            )
            # Load robot
            f_path_robot_urdf = os.path.join(self.urdf_pkg_path, self.cfg.robot_urdf)
            self.robot_id = self.bullet_client.loadURDF(
                fileName=f_path_robot_urdf,
                basePosition=self.robot_pos,
                baseOrientation=self.robot_ori
                )
            # Load socket
            f_path_socket_urdf = os.path.join(self.urdf_pkg_path, self.cfg.socket_urdf)
            self.socket_id = self.bullet_client.loadURDF(
                fileName=f_path_socket_urdf,
                basePosition=self.socket_pos,
                baseOrientation=self.socket_ori
            )
            # Set gravity
            self.bullet_client.setGravity(*self.cfg.gravity)
            # Create bullet body helper objects
            self.ur_arm.connect(self.bullet_client, self.robot_id, enable_fts=True)
            self.socket.connect(self.bullet_client, self.socket_id)

        self.ur_arm.reset(joint_cfg=joint_conf)
        self.ur_arm.update()
        self.socket.update()

    def sub_step(self) -> None:
        self.ur_arm.update()
