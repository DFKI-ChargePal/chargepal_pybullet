""" This file defines the UR arm robot class """
# global
from dataclasses import dataclass, field
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.bullet import (
    ARM_LINK_NAMES,
    ARM_JOINT_NAMES,
    ARM_JOINT_LIMITS,
    BulletJointState,
    ARM_JOINT_DEFAULT_VALUES
)
from gym_chargepal.bullet.body_link import BodyLink
from gym_chargepal.bullet.ft_sensor import FTSensor
from gym_chargepal.utility.cfg_handler import ConfigHandler
from gym_chargepal.bullet.utility import create_joint_index_dict

# mypy
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class URArmCfg(ConfigHandler):
    arm_link_names: List[str] = field(default_factory=lambda: ARM_LINK_NAMES)
    arm_joint_names: List[str] = field(default_factory=lambda: ARM_JOINT_NAMES)
    joint_default_values: Dict[str, float] = field(default_factory=lambda: ARM_JOINT_DEFAULT_VALUES)
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: ARM_JOINT_LIMITS)
    tcp_link_name: str = 'plug'
    ft_joint_name: str = 'mounting_to_wrench'


class URArm:
    
    def __init__(self, config: Dict[str, Any]):
        # Create configuration and override values
        self.cfg = URArmCfg()
        self.cfg.update(**config)

    def connect(self, bullet_client: BulletClient, body_id: int, enable_fts: bool=False) -> None:
        # Safe references
        self.bc = bullet_client
        self.body_id = body_id
        self.joint_idx_dict = create_joint_index_dict(
            body_id, 
            self.cfg.arm_joint_names, 
            bullet_client
            )
        self.tcp = BodyLink(self.cfg.tcp_link_name, bullet_client, body_id)
        self.fts = FTSensor(self.cfg.ft_joint_name, bullet_client, body_id) if enable_fts else None

    def reset(self, joint_cfg: Optional[Tuple[float, ...]] = None) -> None:
        """ Hard arm reset in either default or given joint configuration """
        # reset joint configuration
        if joint_cfg is None:
            for joint_name in self.joint_idx_dict.keys():
                self.bc.resetJointState(
                    bodyUniqueId=self.body_id,
                    jointIndex=self.joint_idx_dict[joint_name],
                    targetValue=self.cfg.joint_default_values[joint_name],
                    targetVelocity=0.0
                    )
        else:
            assert len(joint_cfg) == len(self.joint_idx_dict)
            for k, joint_name in enumerate(self.joint_idx_dict.keys()):
                joint_state = joint_cfg[k]
                self.bc.resetJointState(
                    bodyUniqueId=self.body_id,
                    jointIndex=self.joint_idx_dict[joint_name], 
                    targetValue=joint_state,
                    targetVelocity=0.0
                    )

    def update(self) -> None:
        """ Update physical pybullet state """
        self.state = self.bc.getJointStates(
            bodyUniqueId=self.body_id,
            jointIndices=[idx for idx in self.joint_idx_dict.values()]
        )
        if self.fts: self.fts.update()
        self.tcp.update()

    def get_joint_pos(self) -> Tuple[float, ...]:
        state_idx = BulletJointState.JOINT_POSITION
        pos = tuple(joint[state_idx] for joint in self.state)
        return pos

    def get_joint_vel(self) -> Tuple[float, ...]:
        state_idx = BulletJointState.JOINT_VELOCITY
        vel = tuple(joint[state_idx] for joint in self.state)
        return vel
