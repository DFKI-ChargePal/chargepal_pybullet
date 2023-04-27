""" This file defines the PyBullet Socket class. """
# global
from dataclasses import dataclass
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet.body_link import BodyLink
from gym_chargepal.utility.cfg_handler import ConfigHandler

#mypy
from typing import Any, Dict


@dataclass
class SocketCfg(ConfigHandler):
    socket_link_name = "socket"


class Socket:

    def __init__(self, config: Dict[str, Any]):
        # Create configuration and override values
        self.cfg = SocketCfg()
        self.cfg.update(**config)

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
