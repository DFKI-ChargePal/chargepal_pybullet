# global
import numpy as np
from collections import deque
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet import BulletJointState
import gym_chargepal.bullet.utility as pb_utils

# mypy
from typing import Tuple, Deque
from numpy import typing as npt


class FTSensor:

    def __init__(self, joint_name: str, bullet_client: BulletClient, body_id: int, buffer_size: int):
        self.bc = bullet_client
        self.body_id = body_id
        self.joint_name = joint_name
        self.joint_idx = pb_utils.get_joint_idx(
            body_id=body_id,
            joint_name=joint_name,
            bullet_client=bullet_client
        )
        self.buffer: Deque[npt.NDArray[np.float_]] = deque(maxlen=buffer_size)
        self.enable()

    def enable(self) -> None:
        # enable Force-Torque sensor
        self.bc.enableJointForceTorqueSensor(
            bodyUniqueId=self.body_id,
            jointIndex=self.joint_idx,
            enableSensor=True
            )

    def disable(self) -> None:
        # disable Force-Torque sensor
        self.bc.enableJointForceTorqueSensor(
            bodyUniqueId=self.body_id,
            jointIndex=self.joint_idx,
            enableSensor=False
        )

    def update(self) -> None:
        self.state = self.bc.getJointState(
            bodyUniqueId=self.body_id,
            jointIndex=self.joint_idx
        )

    def get_wrench(self) -> Tuple[float, ...]:
        state_idx = BulletJointState.JOINT_REACTION_FORCE
        wrench: Tuple[float, ...] = self.state[state_idx]
        self.buffer.append(np.array(wrench, dtype=np.float32))
        mean_wrench = tuple(np.mean(self.buffer, axis=0, dtype=np.float32).tolist())
        return mean_wrench
