""" This file defines the point to point worlds. """
import os
import copy
import logging

# mypy
from typing import Dict, Any, Tuple, Union

import pybullet as p

from gym_chargepal.worlds.config import WORLD_PTP
from gym_chargepal.worlds.world import World
from gym_chargepal.bullet.utility import (
    draw_cylinder_marker,
    create_joint_index_dict,
    create_link_index_dict,
    get_link_idx,
)

LOGGER = logging.getLogger(__name__)


class WorldPoint2Point(World):

    """ Build a robot worlds where the task is to move from one point to one anther point """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD_PTP)
        config.update(hyperparams)
        World.__init__(self, config)
        self.plane_id = -1
        self.robot_id = -1
        self.target_id = -1

        self.ur_joint_idx_dict: Dict[str, int] = {}
        self.plug_ref_frame_idx_dict: Dict[str, int] = {}
        self.plug_reference_frame_idx: int = -1

        self.ur_joint_start_config: Dict[str, float] = self._hyperparams['ur_joint_start_config']
        # Check if target configuration is initialized
        assert self._hyperparams['target_pos']
        assert self._hyperparams['target_ori']
        self.target_pos: Tuple[float, ...] = self._hyperparams['target_pos']
        self.target_ori: Tuple[float, ...] = self._hyperparams['target_ori']

    def _init_idx(self) -> None:
        if self.physics_client_id < 0:
            error_msg = f'Unable to set link/joint indices! Did you connect with a Bullet physics server?'
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            self.plug_reference_frame_idx = get_link_idx(
                body_id=self.robot_id, 
                link_name=self._hyperparams['plug_reference_frame'],
                client_id=self.physics_client_id
                )
            self.ur_joint_idx_dict = create_joint_index_dict(
                body_id=self.robot_id,
                joint_names=self._hyperparams['ur_joint_names'],
                client_id=self.physics_client_id
                )
            self.plug_ref_frame_idx_dict = create_link_index_dict(
                body_id=self.robot_id,
                link_names=self._hyperparams['plug_ref_frame_names'],
                client_id=self.physics_client_id
                )

    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None, render: bool = False) -> None:

        if self.physics_client_id < 0:
            # connect to bullet simulation server
            self.connect(render)
            # load plane
            self.plane_id = p.loadURDF('plane.urdf', physicsClientId=self.physics_client_id)
            # load robot
            f_path_robot_urdf = os.path.join(self.urdf_pkg_path, self._hyperparams['robot_urdf'])
            self.robot_id = p.loadURDF(
                fileName=f_path_robot_urdf,
                basePosition=self._hyperparams['robot_start_pos'],
                baseOrientation=self._hyperparams['robot_start_ori'], 
                physicsClientId=self.physics_client_id
                )
            # set gravity
            p.setGravity(*self._hyperparams['gravity'], physicsClientId=self.physics_client_id)
            # initialize joint and link indices
            self._init_idx()
            # notify all references about the changes
            self.notify_bullet_obs()

        # reset joint configuration
        if joint_conf is None:
            for joint_name in self.ur_joint_idx_dict.keys():
                p.resetJointState(
                    bodyUniqueId=self.robot_id, 
                    jointIndex=self.ur_joint_idx_dict[joint_name], 
                    targetValue=self.ur_joint_start_config[joint_name],
                    targetVelocity=0.0, 
                    physicsClientId=self.physics_client_id
                    )
        else:
            assert len(joint_conf) == len(self.ur_joint_idx_dict)
            for k, joint_name in enumerate(self.ur_joint_idx_dict.keys()):
                joint_state = joint_conf[k]
                p.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.ur_joint_idx_dict[joint_name], 
                    targetValue=joint_state,
                    targetVelocity=0.0, 
                    physicsClientId=self.physics_client_id
                    )
        # draw target
        self.draw_target(render)

    def draw_target(self, render: bool) -> None:
        if render:
            if self.target_id > -1:
                p.removeBody(self.target_id, physicsClientId=self.physics_client_id)
            self.target_id = draw_cylinder_marker(
                position=self.target_pos, 
                orientation=self.target_ori,
                radius=0.035, 
                height=0.080,
                color=(1, 0, 0, 0.75), 
                physics_client_id=self.physics_client_id
                )
