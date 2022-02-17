""" This file defines the point to point worlds. """
import copy

# mypy
from typing import Dict, Any, Tuple, Union

import pybullet as p
import pybullet_data

from gym_env.worlds.config import WORLD_PTP
from gym_env.worlds.world import World
from gym_env.bullet.utility import draw_sphere_marker


class WorldPoint2Point(World):

    """ Build a robot worlds where the task is to move from one point to one anther point """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD_PTP)
        config.update(hyperparams)
        World.__init__(self, config)
        # load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF('plane.urdf', physicsClientId=self.physics_client_id)
        # load robot base
        self.platform_id = p.loadURDF(self._hyperparams['platform_urdf'], physicsClientId=self.physics_client_id)
        # load robot arm
        self.arm_id = p.loadURDF(self._hyperparams['robot_urdf'], physicsClientId=self.physics_client_id)
        # set gravity
        p.setGravity(*self._hyperparams['gravity'], physicsClientId=self.physics_client_id)
        # constants
        self.tool_frame_idx: int = self._hyperparams['tool_frame_idx']
        self.target: Tuple[float, ...] = self._hyperparams['target']
        if self._hyperparams['gui']:
            self.target_id = draw_sphere_marker(self.target, 0.045, (1, 0, 0, 0.75), self.physics_client_id)

    def reset(self, joint_conf: Union[None, Tuple[float, ...]] = None) -> None:
        # reset joint configuration
        if joint_conf is None:
            for j_idx_key, j_idx_value in self.joint_idx.items():
                j_start_pos = self.joint_x0[j_idx_key]
                p.resetJointState(self.arm_id, j_idx_value, j_start_pos, physicsClientId=self.physics_client_id)
        else:
            assert len(joint_conf) == len(self.joint_idx)
            for i, j_idx in enumerate(self.joint_idx.values()):
                j_state = joint_conf[i]
                p.resetJointState(self.arm_id, j_idx, j_state, physicsClientId=self.physics_client_id)

    def draw_target(self) -> None:
        if self._hyperparams['gui']:
            p.removeBody(self.target_id, physicsClientId=self.physics_client_id)
            self.target_id = draw_sphere_marker(self.target, 0.045, (1, 0, 0, 0.75), self.physics_client_id)
