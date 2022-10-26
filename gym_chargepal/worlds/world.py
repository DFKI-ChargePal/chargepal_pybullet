""" This file defines the worlds base class. """
# global
import os
import abc
import copy
import logging
import time
import rospkg
import pybullet as p
import pybullet_data
from dataclasses import dataclass
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.utility.cfg_handler import ConfigHandler

# mypy
from typing import Dict, Any, Union, Tuple, List, Optional
from gym_chargepal.sensors.sensor import Sensor


LOGGER = logging.getLogger(__name__)

@dataclass
class WorldCfg(ConfigHandler):
    freq_sim: int = 240
    freq_ctrl: int = 60
    gravity: Tuple[float, ...] = (0.0, 0.0, -9.81)
    urdf_model_dir: str = '_bullet_urdf_models'
    model_description_pkg = 'chargepal_description'


class World(metaclass=abc.ABCMeta):
    """ World superclass. """
    def __init__(self, config: Dict[str, Any]):
        # Create configuration and override values
        self.cfg = WorldCfg()
        self.cfg.update(**config)
        self.bullet_client: BulletClient = None
        self.sim_steps = int(self.cfg.freq_sim // self.cfg.freq_ctrl)
        # find chargepal ros description package
        ros_pkg = rospkg.RosPack()
        ros_pkg_path = ros_pkg.get_path(self.cfg.model_description_pkg)
        self.urdf_pkg_path = os.path.join(ros_pkg_path, self.cfg.urdf_model_dir)

    def connect(self, gui: bool) -> None:
        # connecting to bullet server
        connection_mode = p.GUI if gui else p.DIRECT
        self.bullet_client = BulletClient(connection_mode=connection_mode)
        assert self.bullet_client
        # set common bullet data path
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        # disable real-time simulation
        self.bullet_client.setRealTimeSimulation(False)
        # reset simulation
        self.bullet_client.resetSimulation()
        self.bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        if gui:
            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self.bullet_client.resetDebugVisualizerCamera(
                cameraDistance=2.5, 
                cameraYaw=50, 
                cameraPitch=-42,
                cameraTargetPosition=[0,0,1.2]
                )

    def disconnect(self) -> None:
        if self.bullet_client is not None:
            connection_info = self.bullet_client.getConnectionInfo()
            if connection_info['isConnected'] > 0:
                self.bullet_client.disconnect()
            self.bullet_client = None

    def step(self, render: bool, sensors: Optional[List[Sensor]] = None) -> None:
        # step bullet simulation
        if self.bullet_client is None:
            error_msg = f'Unable to step simulation! Did you connect with a Bullet physics server?'
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        for _ in range(self.sim_steps):
            self.bullet_client.stepSimulation()
            # update physics in subclass
            self.sub_step()
            # wait to render in wall clock time
            if render:
                time.sleep(1./self.cfg.freq_sim)

    @abc.abstractmethod
    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None) -> None:
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def sub_step(self) -> None:
        """ The step function of the subclass. This function will be called after each physical simulation step. """
        raise NotImplementedError('Must be implemented in subclass')