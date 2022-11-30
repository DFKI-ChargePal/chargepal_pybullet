""" This file defines the plugger worlds. """
# global
import os 
import logging
import numpy as np

# local
import gym_chargepal.utility.cfg_handler as cfg_helper

from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.socket import Socket
from gym_chargepal.worlds.world import WorldCfg, World
from gym_chargepal.utility.tf import Quaternion, Translation, Pose

# mypy
from typing import Any, Dict, Tuple, Union


LOGGER = logging.getLogger(__name__)

PLANE_POS = Translation(0.0, 0.0, 0.0)
PLANE_ORI = Quaternion()

ROBOT_POS = Translation(0.0, 1.15, 0.0)
ROBOT_ORI = Quaternion()

SOCKET_POS = Translation(0.0, -0.25/2.0, 0.0)
SOCKET_ORI = Quaternion()
SOCKET_ORI.from_euler_angles(0.0, 0.0, np.pi)


class WorldPluggerCfg(WorldCfg):
    plane_urdf: str = 'plane.urdf'
    robot_urdf: str = 'primitive_chargepal_with_fix_plug.urdf'
    socket_urdf: str = 'primitive_adapter_station.urdf'
    plane_config: Pose = Pose(PLANE_POS, PLANE_ORI)
    robot_config: Pose = Pose(ROBOT_POS, ROBOT_ORI)
    socket_config: Pose = Pose(SOCKET_POS, SOCKET_ORI)


class WorldPlugger(World):
    """ Build a robot world where the task is to connect some type of plug with some type of socket """
    def __init__(self, config: Dict[str, Any]):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: WorldPluggerCfg = WorldPluggerCfg()
        self.cfg.update(**config)

        # Pre initialize class attributes
        self.plane_id = -1
        self.robot_id = -1
        self.socket_id = -1
        ur_arm_config = cfg_helper.search(config, 'ur_arm')
        self.ur_arm = URArm(ur_arm_config)
        socket_config = cfg_helper.search(config, 'socket')
        self.socket = Socket(socket_config)

        # Extract start configurations
        self.plane_pos = self.cfg.plane_config.pos.as_tuple()
        self.plane_ori = self.cfg.plane_config.ori.as_tuple(order='xyzw')
        self.robot_pos = self.cfg.robot_config.pos.as_tuple()
        self.robot_ori = self.cfg.robot_config.ori.as_tuple(order='xyzw')
        self.socket_pos = self.cfg.socket_config.pos.as_tuple()
        self.socket_ori = self.cfg.socket_config.ori.as_tuple(order='xyzw')

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
            self.socket.connect(self.bullet_client, self.socket_id, enable_fts=True)

        self.ur_arm.reset(joint_cfg=joint_conf)
        self.ur_arm.update()
        self.socket.update()

    def sub_step(self) -> None:
        self.ur_arm.update()
