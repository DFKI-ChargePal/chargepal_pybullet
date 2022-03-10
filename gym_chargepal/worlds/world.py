""" This file defines the worlds base class. """
import os
import abc
import copy
import logging
from select import select
import time
import rospkg

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from gym_chargepal.worlds.config import WORLD


# mypy
from typing import Dict, Any, Union, Tuple, List, Optional
from gym_chargepal.sensors.sensor import Sensor
from gym_chargepal.bullet.bullet_observer import BulletObserver


LOGGER = logging.getLogger(__name__)


class World(object):
    """ World superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD)
        config.update(hyperparams)
        self._hyperparams = config
        self.bullet_client = None
        self.physics_client_id = -1
        self.bullet_observers: List[BulletObserver] = []
        self.gravity = self._hyperparams['gravity']
        self.sim_steps = int(self._hyperparams['hz_sim'] // self._hyperparams['hz_ctrl'])
        # find chargepal ros description package
        ros_pkg = rospkg.RosPack()
        ros_pkg_path = ros_pkg.get_path(self._hyperparams['chargepal_description_pkg'])
        self.urdf_pkg_path = os.path.join(ros_pkg_path, self._hyperparams['urdf_sub_dir'])

    def connect(self, gui: bool) -> None:
        # connecting to bullet server
        connection_mode = p.GUI if gui else p.DIRECT
        self.bullet_client = bullet_client.BulletClient(connection_mode=connection_mode)
        assert self.bullet_client is not None
        # set common bullet data path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # extract client id
        self.physics_client_id = self.bullet_client._client
        # disable real-time simulation
        p.setRealTimeSimulation(False, physicsClientId=self.physics_client_id)
        # reset simulation
        self.bullet_client.resetSimulation()
        self.bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    def disconnect(self) -> None:
        connection_info = p.getConnectionInfo(1)
        if connection_info['isConnected'] > 0:
            p.disconnect(physicsClientId=self.physics_client_id)
        self.physics_client_id = -1
        self.bullet_client = None
        self.notify_bullet_obs()

    def attach_bullet_obs(self, bullet_obs: BulletObserver) -> None:
        self.bullet_observers.append(bullet_obs)

    def detach_bullet_obs(self, bullet_obs: BulletObserver) -> None:
        self.bullet_observers.remove(bullet_obs)

    def notify_bullet_obs(self) -> None:
        for obs in self.bullet_observers:
            obs.update_bullet_id()

    def step(self, render: bool, sensors: Optional[List[Sensor]] = None) -> None:
        # step bullet simulation
        if self.physics_client_id < 0:
            error_msg = f'Unable to step simulation! Did you connect with a Bullet physics server?'
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        for _ in range(self.sim_steps):
            p.stepSimulation(physicsClientId=self.physics_client_id)
            # update sensor values as needed
            if sensors is not None:
                for sensor in sensors:
                    sensor.update()
            # wait to render in wall clock time
            if render:
                time.sleep(1./self._hyperparams['hz_sim'])

    @abc.abstractmethod
    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None) -> None:
        raise NotImplementedError('Must be implemented in subclass.')
