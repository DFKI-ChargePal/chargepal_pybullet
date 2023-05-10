from __future__ import annotations
# global
import logging
from dataclasses import dataclass
from rigmopy import Pose

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class StartSamplerCfg(ConfigHandler):
    X_tgt2plug: Pose = Pose()
    variance: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class StartSampler:

    def __init__(self, config: dict[str, Any]) -> None:
                # Create configuration and overwrite values
        self.cfg = StartSamplerCfg()
        self.cfg.update(**config)

    @property
    def random_X_tgt2plug(self) -> Pose:
        return self.cfg.X_tgt2plug.random(*self.cfg.variance)
