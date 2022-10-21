""" This file defines the force torque sensors class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.config import BulletJointState
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.sensors.sensor import SensorCfg, Sensor

# mypy
from typing import Dict, Any, Tuple, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class FTSensorCfg(SensorCfg):
    sensor_id: str = 'ft_sensor'
    force_id: str = 'f'
    moment_id: str = 'm'


class FTSensor(Sensor):
    """ Force Torque Sensor. """
    def __init__(self, config: Dict[str, Any], world: WorldPegInHole):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: FTSensorCfg = FTSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.world = world
        # Intern sensors state
        self.sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self.sensor_state = self.world.bullet_client.getJointState(
            bodyUniqueId=self.world.bullet_client,
            jointIndex=self.world.ft_sensor_joint_idx
        )

    def measurement(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self.sensor_state is not None
        # ToDo: There shout be a noisy measurement signal in future versions.
        state_idx = BulletJointState.JOINT_REACTION_FORCE
        meas = self.sensor_state[state_idx]
        return meas
