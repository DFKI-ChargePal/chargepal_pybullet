""" This file defines the link state class """
# global
from pybullet_utils.bullet_client import BulletClient

# local
import gym_chargepal.bullet.utility as pb_utils
from gym_chargepal.bullet import BulletLinkState
from gym_chargepal.utility.tf import Quaternion, Translation, Twist


class BodyLink:

    def __init__(self, name: str, bullet_client: BulletClient, body_id: int):
        self.bc = bullet_client
        self.body_id = body_id
        self.link_name = name
        self.link_idx = pb_utils.get_link_idx(
            body_id=body_id, 
            link_name=name,
            bullet_client=bullet_client
        )
        if self.link_idx < -1:
            raise ValueError(f"Link with name {name} could not be found! ")

    def update(self) -> None:
        self.state = self.bc.getLinkState(
            bodyUniqueId=self.body_id,
            linkIndex=self.link_idx,
            computeLinkVelocity=True,
            computeForwardKinematics=True
        )

    def get_pos(self) -> Translation:
        # make sure to update the sensor state before calling this method
        state_idx = BulletLinkState.WORLD_LINK_FRAME_POS
        pos = Translation(*self.state[state_idx])
        return pos

    def get_ori(self) -> Quaternion:
        # make sure to update the sensor state before calling this method
        state_idx = BulletLinkState.WORLD_LINK_FRAME_ORI
        ori = Quaternion(*self.state[state_idx] + ('xyzw',))  
        return ori

    def get_twist(self) -> Twist:
        lin_vel_state_idx = BulletLinkState.WORLD_LINK_LIN_VEL
        ang_vel_state_idx = BulletLinkState.WORLD_LINK_ANG_VEL
        return Twist(*self.state[lin_vel_state_idx] + self.state[ang_vel_state_idx])
