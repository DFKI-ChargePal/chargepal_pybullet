""" This file defines the socket sensor class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.socket import Socket
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Any, Dict, Optional, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class SocketSensorCfg(SensorCfg):
    sensor_id: str = 'socket_sensor'
    pos_id: str = 'x'
    ori_id: str = 'q'


class SocketSensor(Sensor):
    """ Sensor class for a fake socket observation. """
    def __init__(self, config: Dict[str, Any], socket: Socket):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: SocketSensorCfg = SocketSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.socket = socket

    def get_pos(self) -> Tuple[float, ...]:
        return self.socket.socket.get_pos()

    def get_ori(self) -> Tuple[float, ...]:
        return self.socket.socket.get_ori()
