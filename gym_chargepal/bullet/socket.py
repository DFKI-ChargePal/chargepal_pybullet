""" This file defines the PyBullet Socket class. """
from __future__ import annotations
# global
import logging
from dataclasses import dataclass
from rigmopy import Vector3d, Quaternion, Pose
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet.body_link import BodyLink
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class SocketCfg(ConfigHandler):
    socket_link_name = "socket"


class Socket:

    _CONNECTION_ERROR_MSG = f"No connection to PyBullet. Please fist connect via {__name__}.connect()"

    def __init__(self, config: dict[str, Any]):
        # Create configuration and override values
        self.cfg = SocketCfg()
        self.cfg.update(**config)

    @property
    def is_connected(self) -> bool:
        """ Check if object is connected to PyBullet server

        Returns:
            Boolean
        """
        return True if self.bc else False

    def connect(self, bullet_client: BulletClient, body_id: int) -> None:
        # Safe references
        self.bc = bullet_client
        self.body_id = body_id
        self.socket = BodyLink(
            name=self.cfg.socket_link_name, 
            bullet_client=bullet_client, 
            body_id=body_id
        )

    def update(self) -> None:
        """ Update physical pybullet state. """
        self.socket.update()

    def get_X_world2socket(self) -> Pose:
        if self.is_connected:
            # Get socket pose
            X_world2socket = self.socket.get_X_world2link()  # type: ignore
        else:
            LOGGER.error(self._CONNECTION_ERROR_MSG)
            X_world2socket = Pose()
        return X_world2socket

    def get_p_world2socket(self) -> Vector3d:
        return self.get_X_world2socket().p

    def get_q_world2socket(self) -> Quaternion:
        return self.get_X_world2socket().q
