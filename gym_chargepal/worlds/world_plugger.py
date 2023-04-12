""" This file defines the plugger worlds. """
# global
import os 
import logging
import numpy as np
from rigmopy import Pose, Quaternion, Vector3d

# local
from gym_chargepal.bullet.ur_arm import URArm
import gym_chargepal.utility.cfg_handler as ch
from gym_chargepal.bullet.socket import Socket
from gym_chargepal.worlds.world import WorldCfg, World

# mypy
from typing import Any, Dict, Tuple, Union


LOGGER = logging.getLogger(__name__)


class WorldPluggerCfg(WorldCfg):
    plane_urdf: str = 'plane.urdf'
    robot_urdf: str = 'primitive_chargepal_with_fix_plug.urdf'
    socket_urdf: str = 'primitive_adapter_station.urdf'
    plane_config: Pose = Pose()
    robot_config: Pose = Pose().from_pq(p=Vector3d().from_xyz((0.0, 1.15, 0.0)))
    socket_config: Pose = Pose().from_pq(p=Vector3d().from_xyz((0.0, -0.25/2.0, 0.0)), q=Quaternion().from_euler_angle(angles=(0.0, 0.0, np.pi)))


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
        ur_arm_config = ch.search(config, 'ur_arm')
        ur_arm_config['ft_buffer_size'] = self.sim_steps + 1
        self.ur_arm = URArm(ur_arm_config)
        socket_config = ch.search(config, 'socket')
        self.socket = Socket(socket_config)
        # Extract start configurations
        self.plane_pos, self.plane_ori = self.cfg.plane_config.xyz_xyzw
        self.robot_pos, self.robot_ori = self.cfg.robot_config.xyz_xyzw
        self.socket_pos, self.socket_ori = self.cfg.socket_config.xyz_xyzw

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
