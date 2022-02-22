""" This file defines the worlds base class. """
import abc
import copy
import time

import pybullet as p
from pybullet_utils import bullet_client


# mypy
from typing import Dict, Any, Union, Tuple, List, Optional


from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.worlds.config import BulletLinkState, BulletJointState

from gym_chargepal.worlds.config import WORLD


class World(object):
    """ World superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD)
        config.update(hyperparams)
        self._hyperparams = config
        # connecting to bullet server
        cm = p.GUI if self._hyperparams['gui'] else p.DIRECT
        self.bullet_client = bullet_client.BulletClient(connection_mode=cm)
        self.physics_client_id = self.bullet_client._client
        self.gravity = self._hyperparams['gravity']
        self.sim_steps = int(self._hyperparams['hz'] // self._hyperparams['hz_env'])
        self.joint_idx: Dict[str, int] = self._hyperparams['joint_idx']
        self.joint_x0: Dict[str, float] = self._hyperparams['joint_x0']
        self.link_state_idx: BulletLinkState = self._hyperparams['link_state_idx']
        self.joint_state_idx: BulletJointState = self._hyperparams['joint_state_idx']
        # disable real-time simulation
        p.setRealTimeSimulation(False, physicsClientId=self.physics_client_id)

    def step(self, sensors: Optional[List[Sensor]] = None) -> None:
        # step bullet simulation
        for _ in range(self.sim_steps):
            p.stepSimulation(physicsClientId=self.physics_client_id)
            # update sensor values as needed
            if sensors is not None:
                for sensor in sensors:
                    sensor.update()
            # wait to render in wall clock time
            if self._hyperparams['gui']:
                time.sleep(1./self._hyperparams['hz'])

    def exit(self) -> None:
        p.disconnect(physicsClientId=self.physics_client_id)
        del self.bullet_client

    @abc.abstractmethod
    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None) -> None:
        raise NotImplementedError('Must be implemented in subclass.')
