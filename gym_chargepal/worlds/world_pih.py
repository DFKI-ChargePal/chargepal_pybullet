""" This file defines the peg in hole in a cartesian controlled world. """
import os
import copy
import logging

# mypy
from typing import Dict, Any, Union, Tuple

from gym_chargepal.worlds.config import WORLD_PIH
from gym_chargepal.worlds.world import World
from gym_chargepal.bullet.utility import (
    create_joint_index_dict,
    create_link_index_dict,
    get_joint_idx,
    get_link_idx,
)


LOGGER = logging.getLogger(__name__)


class WorldPegInHole(World):

    """ Build a world where the task is to plug in a peg. """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD_PIH)
        config.update(hyperparams)
        World.__init__(self, config)
        self.plane_id = -1
        self.robot_id = -1
        self.adpstd_id = -1

        # links and joints
        self.ur_joint_idx_dict: Dict[str, int] = {}
        self.plug_ref_frame_idx_dict: Dict[str, int] = {}
        self.adpstd_ref_frame_idx_dict: Dict[str, int] = {}
        self.ft_sensor_joint_idx: int = -1
        self.plug_reference_frame_idx: int = -1
        self.adpstd_reference_frame_idx: int = -1

        self.ur_joint_start_config: Dict[str, float] = self._hyperparams['ur_joint_start_config']

    def _init_idx(self) -> None:
        if self.bullet_client is None:
            error_msg = f'Unable to set link/joint indices! Did you connect with a Bullet physics server?'
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            self.ft_sensor_joint_idx = get_joint_idx(
                body_id=self.robot_id, 
                joint_name=self._hyperparams['ft_sensor_joint'],
                bullet_client=self.bullet_client
                )
            self.plug_reference_frame_idx = get_link_idx(
                body_id=self.robot_id, 
                link_name=self._hyperparams['plug_reference_frame'],
                bullet_client=self.bullet_client
                )
            self.adpstd_reference_frame_idx = get_link_idx(
                body_id=self.adpstd_id, 
                link_name=self._hyperparams['adpstd_reference_frame'],
                bullet_client=self.bullet_client
                )
            self.ur_joint_idx_dict = create_joint_index_dict(
                body_id=self.robot_id,
                joint_names=self._hyperparams['ur_joint_names'],
                bullet_client=self.bullet_client
                )
            self.plug_ref_frame_idx_dict = create_link_index_dict(
                body_id=self.robot_id,
                link_names=self._hyperparams['plug_ref_frame_names'],
                bullet_client=self.bullet_client
                )
            self.adpstd_ref_frame_idx_dict = create_link_index_dict(
                body_id=self.adpstd_id,
                link_names=self._hyperparams['adpstd_ref_frame_names'],
                bullet_client=self.bullet_client
                )


    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None, render: bool = False) -> None:

        if self.bullet_client is None:
            # connect to bullet simulation server
            self.connect(render)
            assert self.bullet_client
            # load plane
            self.plane_id = self.bullet_client.loadURDF('plane.urdf')
            # load robot
            f_path_robot_urdf = os.path.join(self.urdf_pkg_path, self._hyperparams['robot_urdf'])
            self.robot_id = self.bullet_client.loadURDF(
                fileName=f_path_robot_urdf,
                basePosition=self._hyperparams['robot_start_pos'],
                baseOrientation=self._hyperparams['robot_start_ori']
                )
            # load adapter station
            f_path_adpstd_urdf = os.path.join(self.urdf_pkg_path, self._hyperparams['adapter_station_urdf'])
            self.adpstd_id = self.bullet_client.loadURDF(
                fileName=f_path_adpstd_urdf, 
                basePosition=self._hyperparams['adpstd_start_pos'],
                baseOrientation=self._hyperparams['adpstd_start_ori']
                )
            # set gravity
            self.bullet_client.setGravity(*self._hyperparams['gravity'])
            # initialize joint and link indices
            self._init_idx()
            # enable Force-Torque Sensor
            self.bullet_client.enableJointForceTorqueSensor(
                bodyUniqueId=self.robot_id,
                jointIndex=self.ft_sensor_joint_idx,
                enableSensor=True
                )

        # reset joint configuration
        if joint_conf is None:
            for joint_name in self.ur_joint_idx_dict.keys():
                self.bullet_client.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.ur_joint_idx_dict[joint_name],
                    targetValue=self.ur_joint_start_config[joint_name],
                    targetVelocity=0.0
                    )
        else:
            assert len(joint_conf) == len(self.ur_joint_idx_dict)
            for k, joint_name in enumerate(self.ur_joint_idx_dict.keys()):
                joint_state = joint_conf[k]
                self.bullet_client.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.ur_joint_idx_dict[joint_name], 
                    targetValue=joint_state,
                    targetVelocity=0.0
                    )
