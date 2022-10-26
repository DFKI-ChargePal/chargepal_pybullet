# global
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet import BulletJointState
import gym_chargepal.bullet.utility as pb_utils

# mypy
from typing import Tuple


class FTSensor:

    def __init__(self, joint_name: str, bullet_client: BulletClient, body_id: int):
        self.bc = bullet_client
        self.body_id = body_id
        self.joint_name = joint_name
        self.joint_idx = pb_utils.get_joint_idx(
            body_id=body_id,
            joint_name=joint_name,
            bullet_client=bullet_client
        )
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
        return wrench
