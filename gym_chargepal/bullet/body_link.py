""" This file defines the link state class """
from __future__ import annotations
# global
from rigmopy import Quaternion, Vector3d, Vector6d, Pose
from pybullet_utils.bullet_client import BulletClient

# local
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.bullet import BulletLinkState, BulletDynamicsInfo

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
        self.__dyn_info = None
        if self.idx < 0:
            raise ValueError(f"Link with name {name} could not be found! ")

    def update(self) -> None:
        self.__state = self.bc.getLinkState(
            bodyUniqueId=self.body_id,
            linkIndex=self.idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True
        )
        if self.__dyn_info is None:
            self.__dyn_info = self.bc.getDynamicsInfo(
                bodyUniqueId=self.body_id,
                linkIndex=self.idx,
            )

    @property
    def raw_state(self) -> tuple[tuple[float, ...], ...]:
        if self.__state:
            return self.__state
        else:
            raise ValueError("Link state is not set. Please update the link before access state.")

    @property
    def dynamics_info(self) -> tuple[Any, ...]:
        if self.__dyn_info:
            return self.__dyn_info
        else:
            raise ValueError("Link dynamics info is not set. Please update the link before access dynamics info.")

    # //////////////////////////////////////////////////////////////////////// #
    # /// make sure to update the sensor state before calling this methods /// #
    # //////////////////////////////////////////////////////////////////////// #
    @property
    def p_world2link(self) -> Vector3d:
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        return Vector3d().from_xyz(self.raw_state[state_idx])

    @property
    def q_world2link(self) -> Quaternion:
        # make sure to update the sensor state before calling this method
        state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        return Quaternion().from_xyzw(self.raw_state[state_idx])

    @property
    def X_world2link(self) -> Pose:
        return Pose().from_pq(self.p_world2link, self.q_world2link)

    @property
    def p_link2inertial(self) -> Vector3d:
        state_idx = BulletLinkState.LOCAL_INERTIAL_POS
        return Vector3d().from_xyz(self.raw_state[state_idx])

    @property
    def q_link2inertial(self) -> Quaternion:
        state_idx = BulletLinkState.LOCAL_INERTIAL_ORI
        return Quaternion().from_xyzw(self.raw_state[state_idx])

    @property
    def twist_world2link(self) -> Vector6d:
        lin_vel_state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        ang_vel_state_idx = BulletLinkState.WORLD_LINK_ANG_VEL
        return Vector6d().from_xyzXYZ(self.raw_state[lin_vel_state_idx] + self.raw_state[ang_vel_state_idx])

    @property
    def mass(self) -> float:
        m: float = self.dynamics_info[BulletDynamicsInfo.MASS]
        return m
