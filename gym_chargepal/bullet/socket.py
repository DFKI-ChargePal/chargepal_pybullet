""" This file defines the PyBullet Socket class. """
from __future__ import annotations
# global
import logging
import numpy as np
from dataclasses import dataclass
from rigmopy import Vector3d, Quaternion, Pose
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet.ur_arm import URArm
from gym_chargepal.bullet.body_link import BodyLink
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class SocketCfg(ConfigHandler):
    link_name = "socket"
    X_socket2base: Pose = Pose(). from_xyz((0.0, 0.0, 0.05)).from_euler_angle((np.pi, 0.0, 0.0)).inverse()
    X_arm2socket: Pose = Pose().from_xyz((0.635, 0.319, 0.271)).from_euler_angle((0.0, np.pi/2, 0.0))


class Socket:

    _CONNECTION_ERROR_MSG = f"No connection to PyBullet. Please fist connect via {__name__}.connect()"

    def __init__(self, config: dict[str, Any], ur_arm: URArm):
        # Create configuration and override values
        self.cfg = SocketCfg()
        self.cfg.update(**config)
        self.ur_arm = ur_arm
        # PyBullet references
        self._bc: BulletClient | None = None
        self._body_id: int | None = None
        self._socket: BodyLink | None = None

    @property
    def is_connected(self) -> bool:
        """ Check if object is connected to PyBullet server

        Returns:
            Boolean
        """
        return True if self._bc else False

    @property
    def bullet_client(self) -> BulletClient:
        if self.is_connected:
            return self._bc
        else:
            raise RuntimeError("Not connected to PyBullet client.")
        
    @property
    def bullet_body_id(self) -> int:
        if self.is_connected and self._body_id:
            return self._body_id
        else:
            raise RuntimeError("Not connected to PyBullet client.")
        
    @property
    def socket(self) -> BodyLink:
        if self.is_connected and self._socket:
            return self._socket
        else:
            raise RuntimeError("Not connected to PyBullet client.")

    def connect(self, bullet_client: BulletClient, body_id: int) -> None:
        # Safe references
        self._bc = bullet_client
        self._body_id = body_id
        self._socket = BodyLink(
            name=self.cfg.link_name, 
            bullet_client=bullet_client, 
            body_id=body_id
        )

    def update(self) -> None:
        """ Update physical pybullet state. """
        self.socket.update()

    @property
    def X_arm2socket(self) -> Pose:
        if self.is_connected:
            # Get socket pose
            X_arm2socket = self.ur_arm.X_world2arm.inverse() * self.socket.X_world2link
        else:
            LOGGER.error(self._CONNECTION_ERROR_MSG)
            X_arm2socket = Pose()
        return X_arm2socket

    @property
    def p_arm2socket(self) -> Vector3d:
        return self.X_arm2socket.p

    @property
    def q_arm2socket(self) -> Quaternion:
        return self.X_arm2socket.q
