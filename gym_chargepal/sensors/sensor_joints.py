""" This file defines the sensors joint class. """
# global
import logging
from dataclasses import dataclass

# local
from gym_chargepal.bullet.config import BulletJointState
from gym_chargepal.worlds.world_pih import WorldPegInHole
from gym_chargepal.sensors.sensor import SensorCfg, Sensor
from gym_chargepal.worlds.world_ptp import WorldPoint2Point

# mypy
from typing import Dict, Any, List, Tuple, Union, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class JointSensorCfg(SensorCfg):
    sensor_id: str = 'joint_sensor'
    pos_id: str = 'joint_pos'
    vel_id: str = 'joint_vel'
    acc_id: str = 'joint_acc'


class JointSensor(Sensor):
    """ Sensor of the arm joints. """
    def __init__(self, config: Dict[str, Any], world: Union[WorldPoint2Point, WorldPegInHole]):
        # Call super class
        super().__init__(config=config)
        # Create configuration and override values
        self.cfg: JointSensorCfg = JointSensorCfg()
        self.cfg.update(**config)
        # Safe references
        self.world = world
        # intern sensors state
        self._sensor_state: Optional[Tuple[Tuple[float, ...], ...]] = None

    def update(self) -> None:
        self._sensor_state = self.world.bullet_client.getJointStates(
            bodyUniqueId=self.world.robot_id,
            jointIndices=[idx for idx in self.world.ur_joint_idx_dict.values()]
        )

    def get_pos(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletJointState.JOINT_POSITION
        pos = tuple(joint[state_idx] for joint in self._sensor_state)
        return pos

    def get_vel(self) -> Tuple[float, ...]:
        # make sure to update the sensor state before calling this method
        assert self._sensor_state is not None
        state_idx = BulletJointState.JOINT_VELOCITY
        vel = tuple(joint[state_idx] for joint in self._sensor_state)
        return vel
