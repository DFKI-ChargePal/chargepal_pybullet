""" This file defines the peg in hole in a cartesian control worlds. """
import copy

# mypy
from typing import Dict, Any, Union, Tuple

import pybullet as p
import pybullet_data

from gym_chargepal.worlds.config import WORLD_PIH
from gym_chargepal.worlds.world import World


class WorldPegInHole(World):

    """ Build a worlds where the task is to plug in a peg. """
    def __init__(self, hyperparams: Dict[str, Any]):
        config: Dict[str, Any] = copy.deepcopy(WORLD_PIH)
        config.update(hyperparams)
        World.__init__(self, config)
        # load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF('plane.urdf', physicsClientId=self.physics_client_id)
        # load robot base
        self.platform_id = p.loadURDF(self._hyperparams['platform'], physicsClientId=self.physics_client_id)
        # load pillar
        self.pillar_id = p.loadURDF(self._hyperparams['pillar'], physicsClientId=self.physics_client_id)
        # load arm
        self.arm_id = p.loadURDF(self._hyperparams['arm'], physicsClientId=self.physics_client_id)
        # set gravity
        p.setGravity(*self._hyperparams['gravity'], physicsClientId=self.physics_client_id)
        # constants
        self.tool_frame_idx: int = self._hyperparams['tool_frame_idx']
        self.ft_sensor_idx: int = self._hyperparams['ft_sensor_idx']
        self.target_frame_idx: int = self._hyperparams['target_frame_idx']
        self.tool_virtual_frame_idx: int = self._hyperparams['virtual_frame_idx']['tool']
        self.target_virtual_frame_idx: int = self._hyperparams['virtual_frame_idx']['target']
        # enable Force-Torque Sensor
        p.enableJointForceTorqueSensor(self.arm_id, self.ft_sensor_idx, True, physicsClientId=self.physics_client_id)

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
