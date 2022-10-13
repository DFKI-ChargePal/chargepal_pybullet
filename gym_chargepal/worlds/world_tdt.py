""" This file defines the top down experiment task in a cartesian controlled world. """
# global
import os
import copy 
import logging

# local
from gym_chargepal.worlds.world import World
from gym_chargepal.worlds.config import WORLD_TDT
import gym_chargepal.bullet.utility as bullet_helper


# mypy
from typing import Any, Dict, Union, Tuple


LOGGER = logging.getLogger(__name__)


class WorldTopDownTask(World):

    """ Build a testbed with a top down peg-in-hole task. """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD_TDT)
        config.update(hyperparams)
        super().__init__(config)

        # pybullet model ids
        self.plane_id = -1
        self.robot_id = -1
        self.socket_id = -1

        # links and joints
        self.ur_joint_idx_dict: Dict[str, int] = {}
        self.ft_sensor_joint_idx: int = -1
        self.plug_reference_frame_idx: int = -1
        self.socket_frame_idx: int = -1

        self.ur_joint_start_config: Dict[str, float] = self._hyperparams['ur_joint_start_config']

    def _init_idx(self) -> None:
        if self.bullet_client is None:
            error_msg = f'Unable to set link/joint indices! Did you connect with a Bullet physics server?'
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            self.ur_joint_idx_dict = bullet_helper.create_joint_index_dict(
                body_id=self.robot_id,
                joint_names=self._hyperparams['ur_joint_names'],
                bullet_client=self.bullet_client
                )
            self.ft_sensor_joint_idx = bullet_helper.get_joint_idx(
                body_id=self.robot_id, 
                joint_name=self._hyperparams['ft_sensor_joint'],
                bullet_client=self.bullet_client
            )
            self.plug_reference_frame_idx = bullet_helper.get_link_idx(
                body_id=self.robot_id, 
                link_name=self._hyperparams['plug_link_name'],
                bullet_client=self.bullet_client
            )
            self.socket_frame_idx = bullet_helper.get_link_idx(
                body_id=self.socket_id,
                link_name=self._hyperparams['socket_link_name'],
                bullet_client=self.bullet_client
            )

    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None, render: bool = False) -> None:
        
        if self.bullet_client is None:
             # connect to bullet simulation server
            self.connect(render)
            assert self.bullet_client
            # load plane
            self.plane_id = self.bullet_client.loadURDF(
                fileName='plane.urdf',
                basePosition=self._hyperparams['plane_start_pos'],
                baseOrientation=self._hyperparams['plane_start_ori']
                )
            # load robot
            f_path_robot_urdf = os.path.join(self.urdf_pkg_path, self._hyperparams['robot_urdf'])
            self.robot_id = self.bullet_client.loadURDF(
                fileName=f_path_robot_urdf,
                basePosition=self._hyperparams['robot_start_pos'],
                baseOrientation=self._hyperparams['robot_start_ori']
                )
            # load socket
            f_path_socket_urdf = os.path.join(self.urdf_pkg_path, self._hyperparams['socket_urdf'])
            self.socket_id = self.bullet_client.loadURDF(
                fileName=f_path_socket_urdf, 
                basePosition=self._hyperparams['socket_start_pos'],
                baseOrientation=self._hyperparams['socket_start_ori']
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
            # notify all references about the changes
            self.notify_bullet_obs()
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
