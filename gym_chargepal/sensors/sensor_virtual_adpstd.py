""" This file defines the virtual target sensor class. """
# global
import copy
import logging

# local
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.sensors.config import VIRTUAL_ADAPTER_STATION_SENSOR
from gym_chargepal.bullet.config import BulletLinkState

# mypy
from typing import Dict, Any, Tuple, List, Optional
from gym_chargepal.worlds.world_pih import WorldPegInHole


LOGGER = logging.getLogger(__name__)


class VirtualAdapterStationSensor(Sensor):
    """ Sensor of the virtual target frames. """
    def __init__(self, hyperparams: Dict[str, Any], world: WorldPegInHole):
        config = copy.deepcopy(VIRTUAL_ADAPTER_STATION_SENSOR)
        config.update(hyperparams)
        Sensor.__init__(self, config)
        # params
        self.world = world
        # intern sensors state
        self.sensor_state: Optional[Tuple[Tuple[Tuple[float, ...], ...], ...]] = None

    def update(self) -> None:
        self.sensor_state = self.world.bullet_client.getLinkStates(
            bodyUniqueId=self.world.adpstd_id,
            linkIndices=[idx for idx in self.world.adpstd_ref_frame_idx_dict.values()],
            computeLinkVelocity=True,
            computeForwardKinematics=True
        )

    def get_pos_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        pos_list = [state[state_idx] for state in self.sensor_state]
        return pos_list

    def get_vel_list(self) -> List[Tuple[float, ...]]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        pos_list: List[Tuple[float, ...]] = [state[state_idx] for state in self.sensor_state]
        return pos_list
