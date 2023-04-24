""" This file defines the PyBullet reference link class. """
from __future__ import annotations

# global
from dataclasses import dataclass
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet.body_link import BodyLink
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any


@dataclass
class ReferenceBodyCfg(ConfigHandler):
    ref_link_name = "base_link"


class ReferenceBody:

    def __init__(self, config: dict[str, Any]) -> None:
        # Create configuration and override values
        self.cfg = ReferenceBodyCfg()
        self.cfg.update(**config)
    
    def connect(self, bullet_client: BulletClient, body_id: int) -> None:
        # Safe references
        self.bc = bullet_client
        self.body_id = body_id
        self.link = BodyLink(
            name=self.cfg.ref_link_name,
            bullet_client=bullet_client,
            body_id=body_id
        )

    def update(self) -> None:
        """ Update physical pybullet state. """
        self.link.update()
