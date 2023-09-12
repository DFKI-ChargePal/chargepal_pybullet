""" This file defines the plugger worlds. """
from __future__ import annotations

# global
import logging
import pybullet as p
from rigmopy import Pose

# local
from gym_chargepal.bullet.socket import Socket
from gym_chargepal.worlds.world import WorldCfg, World
from gym_chargepal.utility.start_sampler import StartSampler

# mypy
from typing import Any


LOGGER = logging.getLogger(__name__)

_TABLE_HEIGHT = 0.8136


class WorldPluggerCfg(WorldCfg):
    socket_urdf: str = 'tdt_socket.urdf'
    plane_config: Pose = Pose().from_xyz((0.0, 0.0, -_TABLE_HEIGHT))
    env_config: Pose = Pose()


class WorldPlugger(World):

    def __init__(self, 
                 config:        dict[str, Any], 
                 config_arm:    dict[str, Any], 
                 config_start:  dict[str, Any],
                 config_socket: dict[str, Any]
                 ) -> None:
        """ Build a robot world where the task is to connect some type of plug with some type of socket 

        Args:
            config:        Dictionary to overwrite values of the world configuration class
            config_arm:    Dictionary to overwrite values of the UR arm configuration class
            config_start:  Dictionary to overwrite values of the start sampler configuration class
            config_socket: Dictionary to overwrite values of the socket configuration class
        """
        # Call super class
        config_arm['ft_enable'] = True
        super().__init__(config=config, config_arm=config_arm)
        # Create configuration and override values
        self.cfg: WorldPluggerCfg = WorldPluggerCfg()
        self.cfg.update(**config)
        # Pre initialize class attributes
        self.env_id = -1
        self.plane_id = -1
        self.robot_id = -1
        self.socket_id = -1
        self.start = StartSampler(config_start)
        self.socket = Socket(config_socket, self.ur_arm)

    def sample_X0(self) -> Pose:
        X0_world2plug = self.ur_arm.X_world2arm * self.socket.X_arm2socket * self.start.random_X_tgt2plug  * self.ur_arm.cfg.tcp_link_offset.inverse()
        return X0_world2plug

    def reset(self, joint_conf: tuple[float, ...] | None = None, render: bool = False) -> None:
        # Call base class reset method
        super().reset(joint_conf, render)
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
            # Load socket
            socket_pos, socket_ori = (
                self.ur_arm.X_world2arm * self.socket.cfg.X_arm2socket * self.socket.cfg.X_socket2base).xyz_xyzw
            self.socket_id = self.bullet_client.loadURDF(
                fileName=str(self.urdf_pkg_path.joinpath(self.cfg.socket_urdf)),
                basePosition=socket_pos,
                baseOrientation=socket_ori
            )
            # Set gravity
            self.bullet_client.setGravity(*self.cfg.gravity)
            # Create bullet body helper objects
            self.ur_arm.connect(self.bullet_client, self.robot_id)
            self.socket.connect(self.bullet_client, self.socket_id)

        self.ur_arm.reset(joint_cfg=joint_conf)
        self.ur_arm.update()
        self.socket.update()

    def sub_step(self) -> None:
        self.ur_arm.update()
