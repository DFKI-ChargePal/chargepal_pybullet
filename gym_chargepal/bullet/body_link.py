""" This file defines the link state class """
from __future__ import annotations
# global
from rigmopy import Quaternion, Vector3d, Vector6d, Pose
from pybullet_utils.bullet_client import BulletClient

# local
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.bullet import BulletLinkState

# typing
from typing import Any


class BodyLink:

    def __init__(self, name: str, bullet_client: BulletClient, body_id: int):
        self.bc = bullet_client
        self.body_id = body_id
        self.name = name
        self.idx = pb_utils.get_link_idx(
            body_id=body_id, 
            link_name=name,
            bullet_client=bullet_client
        )
        self.__state = None
        if self.idx < 0:
            raise ValueError(f"Link with name {name} could not be found! ")

    def update(self) -> None:
        self.__state = self.bc.getLinkState(
            bodyUniqueId=self.body_id,
            linkIndex=self.idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True
        )

    @property
    def raw_state(self) -> tuple[tuple[float, ...], ...]:
        if self.__state:
            return self.__state
        else:
            raise ValueError("Link state is not set. Please update the link before access state.")

    # //////////////////////////////////////////////////////////////////////// #
    # /// make sure to update the sensor state before calling this methods /// #
    # //////////////////////////////////////////////////////////////////////// #
    def get_p_world2link(self) -> Vector3d:
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        return Vector3d().from_xyz(self.raw_state[state_idx])

    def get_q_world2link(self) -> Quaternion:
        # make sure to update the sensor state before calling this method
        state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        return Quaternion().from_xyzw(self.raw_state[state_idx])

    def get_X_world2link(self) -> Pose:
        return Pose().from_pq(self.get_p_world2link(), self.get_q_world2link())

    def get_p_link2inertial(self) -> Vector3d:
        state_idx = BulletLinkState.LOCAL_INERTIAL_POS
        return Vector3d().from_xyz(self.raw_state[state_idx])

    def get_q_link2inertial(self) -> Quaternion:
        state_idx = BulletLinkState.LOCAL_INERTIAL_ORI
        return Quaternion().from_xyzw(self.raw_state[state_idx])

    def get_twist_world2link(self) -> Vector6d:
        lin_vel_state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        ang_vel_state_idx = BulletLinkState.WORLD_LINK_ANG_VEL
        return Vector6d().from_xyzXYZ(self.raw_state[lin_vel_state_idx] + self.raw_state[ang_vel_state_idx])
