""" This file defines the worlds base class. """
# global
import os
import abc
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
    freq_ctrl: int = 40
    gravity: Tuple[float, ...] = (0.0, 0.0, -9.81)
    urdf_model_dir: str = '_bullet_urdf_models'
    model_description_pkg = 'chargepal_description'
    # Gui configurations
    gui_width: int = 1280
    gui_height: int = 720
    cam_distance: float = 0.75
    cam_yaw: float = 105.0
    cam_pitch: float = -15.0
    cam_x: float = 0.8
    cam_y: float = 0.8
    cam_z: float = 0.15
    # Gui text
    gui_txt: str = ""
    gui_txt_size: float = 5.0
    gui_txt_pos: Tuple[float, ...] = (0.0, 0.0, 0.0)
    gui_txt_rgb: Tuple[float, ...] = (1.0, 1.0, 1.0)
    # Record video stream
    record: bool = False
    rec_file_name: str = "exp_record.mp4"
    rec_fps: int = 240


class World(metaclass=abc.ABCMeta):
    """ World superclass. """
    def __init__(self, config: Dict[str, Any]):
        # Create configuration and override values
        self.cfg = WorldCfg()
        self.cfg.update(**config)
        self.bullet_client: BulletClient = None
        self.sim_steps = int(self.cfg.freq_sim // self.cfg.freq_ctrl)
        # Find chargepal ros description package
        ros_pkg = rospkg.RosPack()
        ros_pkg_path = ros_pkg.get_path(self.cfg.model_description_pkg)
        self.urdf_pkg_path = os.path.join(ros_pkg_path, self.cfg.urdf_model_dir)

    def connect(self, gui: bool) -> None:
        # Connecting to bullet server
        connection_mode = p.GUI if gui else p.DIRECT
        if gui:
            # Add GUI options
            width_opt = f"--width={self.cfg.gui_width}"
            height_opt = f"--height={self.cfg.gui_height}"
            connection_opt = f"{width_opt} {height_opt}"
            if self.cfg.record:
                rec_file_opt = f"--mp4=\"{self.cfg.rec_file_name}\""
                rec_fps_opt = f"--mp4fps={self.cfg.rec_fps}"
                connection_opt = f"{width_opt} {height_opt} {rec_file_opt} {rec_fps_opt}"
        else:
            connection_opt = ""
        self.bullet_client = BulletClient(connection_mode=connection_mode, options=connection_opt)
        assert self.bullet_client
        # Set common bullet data path
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Disable real-time simulation
        self.bullet_client.setRealTimeSimulation(False)
        # Reset simulation
        self.bullet_client.resetSimulation()
        self.bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        if gui:
            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self.bullet_client.resetDebugVisualizerCamera(
                cameraDistance=self.cfg.cam_distance, 
                cameraYaw=self.cfg.cam_yaw, 
                cameraPitch=self.cfg.cam_pitch,
                cameraTargetPosition=[self.cfg.cam_x, self.cfg.cam_y, self.cfg.cam_z]
                )
            if len(self.cfg.gui_txt) >= 0:
                self.bullet_client.addUserDebugText(
                    text=self.cfg.gui_txt,
                    textPosition=self.cfg.gui_txt_pos,
                    textColorRGB=self.cfg.gui_txt_rgb,
                    textSize=self.cfg.gui_txt_size
                )

    def disconnect(self) -> None:
        if self.bullet_client is not None:
            connection_info = self.bullet_client.getConnectionInfo()
            if connection_info['isConnected'] > 0:
                self.bullet_client.disconnect()
            self.bullet_client = None

    def step(self, render: bool, sensors: Optional[List[Sensor]] = None) -> None:
        # Step bullet simulation
        if self.bullet_client is None:
            error_msg = f'Unable to step simulation! Did you connect with a Bullet physics server?'
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        for _ in range(self.sim_steps):
            self.bullet_client.stepSimulation()
            # Update physics in subclass
            self.sub_step()
            # Wait to render in wall clock time
            if render:
                if self.cfg.record:
                    self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                time.sleep(1./self.cfg.freq_sim)

    @abc.abstractmethod
    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None) -> None:
        raise NotImplementedError('Must be implemented in subclass.')

    @abc.abstractmethod
    def sub_step(self) -> None:
        """ The step function of the subclass. This function will be called after each physical simulation step. """
        raise NotImplementedError('Must be implemented in subclass')