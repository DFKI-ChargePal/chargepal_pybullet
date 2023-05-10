# global
import numpy as np
from rigmopy import Vector6d
from collections import deque
from pybullet_utils.bullet_client import BulletClient

# local
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.bullet import BulletJointState

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

    @property
    def wrench(self) -> Vector6d:
        state_idx = BulletJointState.JOINT_REACTION_FORCE
        wrench: Tuple[float, ...] = self.state[state_idx]
        self.buffer.append(np.array(wrench, dtype=np.float64))
        mean_wrench = Vector6d().from_xyzXYZ(np.mean(self.buffer, axis=0, dtype=np.float64))
        return mean_wrench
