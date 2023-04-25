""" This file defines the link state class """
from __future__ import annotations

# global
from rigmopy import Quaternion, Vector3d, Vector6d, Pose
from pybullet_utils.bullet_client import BulletClient

# local
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.bullet import BulletLinkState


class BodyLink:

    def __init__(self, name: str, bullet_client: BulletClient, body_id: int, ref_link: BodyLink | None = None):
        self.bc = bullet_client
        self.body_id = body_id
        self.link_name = name
        self.ref_link = ref_link
        self.link_idx = pb_utils.get_link_idx(
            body_id=body_id, 
            link_name=name,
            bullet_client=bullet_client
        )
        if self.link_idx < -1:
            raise ValueError(f"Link with name {name} could not be found!")

    def update(self) -> None:
        self.link_state = self.bc.getLinkState(
            bodyUniqueId=self.body_id,
            linkIndex=self.link_idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True
        )

    def get_pose_ref(self) -> Pose:
        # Make sure to update the sensor state before calling this method
        pos_state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        ori_state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        # Get reference pose
        p_world2ref = self.ref_link.get_pos_ref() if self.ref_link else Vector3d()
        q_world2ref = self.ref_link.get_ori_ref() if self.ref_link else Quaternion()
        X_world2ref = Pose().from_pq(p_world2ref, q_world2ref)
        # Get target pose
        p_world2tgt = Vector3d().from_xyz(self.link_state[pos_state_idx])
        q_world2tgt = Quaternion().from_xyzw(self.link_state[ori_state_idx])
        X_world2tgt = Pose().from_pq(p_world2tgt, q_world2tgt)
        # Get transformation from reference to target pose
        X_ref2tgt = X_world2ref.inverse() * X_world2tgt
        return X_ref2tgt

    def get_pose_world(self) -> Pose:
        # Make sure to update the sensor state before calling this method
        pos_state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        ori_state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        # Get target pose
        p_world2tgt = Vector3d().from_xyz(self.link_state[pos_state_idx])
        q_world2tgt = Quaternion().from_xyzw(self.link_state[ori_state_idx])
        X_world2tgt = Pose().from_pq(p_world2tgt, q_world2tgt)
        return X_world2tgt
    
    def get_pos_ref(self) -> Vector3d:
        return self.get_pose_ref().p

    def get_pos_world(self) -> Vector3d:
        return self.get_pose_world().p

    def get_ori_world(self) -> Quaternion:
        return self.get_pose_world().q

    def get_ori_ref(self) -> Quaternion:
        return self.get_pose_ref().q

    def get_twist(self) -> Vector6d:
        lin_vel_state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        ang_vel_state_idx = BulletLinkState.WORLD_LINK_ANG_VEL
        return Vector6d().from_xyzXYZ(self.link_state[lin_vel_state_idx] + self.link_state[ang_vel_state_idx])
