""" This file defines the UR arm robot class """
from __future__ import annotations

# global
import logging
import numpy as np
from dataclasses import dataclass, field
from rigmopy import Vector3d, Quaternion, Pose
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
from typing import Any


LOGGER = logging.getLogger(__name__)


_TABLE_WIDTH = 0.81
_PROFILE_SIZE = 0.045
_BASE_PLATE_SIZE = 0.225
_BASE_PLATE_HEIGHT = 0.0225


@dataclass
class URArmCfg(ConfigHandler):
    arm_link_names: list[str] = field(default_factory=lambda: ARM_LINK_NAMES)
    arm_joint_names: list[str] = field(default_factory=lambda: ARM_JOINT_NAMES)
    joint_default_values: dict[str, float] = field(default_factory=lambda: ARM_JOINT_DEFAULT_VALUES)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: ARM_JOINT_LIMITS)
    tcp_link_name: str = 'plug'
    base_link_name: str = 'base'
    ft_joint_name: str = 'mounting_to_wrench'
    ft_buffer_size: int = 1
    X_world2arm: Pose = Pose().from_xyz(
        (_TABLE_WIDTH - _BASE_PLATE_SIZE/2, _PROFILE_SIZE + _BASE_PLATE_SIZE/2, _BASE_PLATE_HEIGHT)
        ).from_euler_angle(angles=(0.0, 0.0, np.pi/2))

class URArm:

    _CONNECTION_ERROR_MSG = f"No connection to PyBullet. Please fist connect via {__name__}.connect()"

    def __init__(self, config: dict[str, Any]) -> None:
        # Create configuration and override values
        self.cfg = URArmCfg()
        self.cfg.update(**config)
        # PyBullet references
        self._bc: BulletClient | None = None
        self._body_id: int | None = None
        # Arm links and joints
        self._base: BodyLink | None = None
        self._tcp: BodyLink | None = None
        self._fts: FTSensor | None = None
        self._joint_idx_dict: dict[str, int] = {}

    @property
    def bullet_client(self) -> BulletClient:
        if self.is_connected:
            return self._bc
        else:
            raise RuntimeError("Not connected to PyBullet client.")

    @property
    def bullet_body_id(self) -> int:
        if self.is_connected and self._body_id:
            return self._body_id
        else:
            raise RuntimeError("Not connected to PyBullet client.")

    @property
    def base_link(self) -> BodyLink:
        if self.is_connected and self._base:
            return self._base
        else:
            raise RuntimeError("Not connected to PyBullet client.")

    @property
    def tcp_link(self) -> BodyLink:
        if self.is_connected and self._tcp:
            return self._tcp
        else:
            raise RuntimeError("Not connected to PyBullet client.")
        
    @property
    def ft_sensor(self) -> FTSensor:
        if self.is_connected and self._fts:
            return self._fts
        else:
            if self.is_connected:
                raise ValueError("F/T sensor is not enabled for this object.")
            else:
                raise RuntimeError("Not connected to PyBullet client.")

    @property
    def is_connected(self) -> bool:
        """ Check if object is connect to PyBullet"""
        return True if self._bc else False

    def connect(self, bullet_client: BulletClient, body_id: int, enable_fts: bool=False) -> None:
        # Safe references
        self._bc = bullet_client
        self._body_id = body_id
        # Map joint names to PyBullet joint ids
        self._joint_idx_dict = create_joint_index_dict(
            body_id,
            self.cfg.arm_joint_names,
            bullet_client
            )
        self._base = BodyLink(
            name=self.cfg.base_link_name,
            bullet_client=bullet_client,
            body_id=body_id
            )
        self._tcp = BodyLink(
            name=self.cfg.tcp_link_name,
            bullet_client=bullet_client,
            body_id=body_id
            )
        if enable_fts:
            self._fts = FTSensor(self.cfg.ft_joint_name, bullet_client, body_id, self.cfg.ft_buffer_size)
        else:
            self._fts = None

    def reset(self, joint_cfg: tuple[float, ...] | None = None) -> None:
        """ Hard arm reset in either default or given joint configuration """
        # reset joint configuration
        if self.is_connected:
            if joint_cfg is None:
                for joint_name in self._joint_idx_dict.keys():
                    self._bc.resetJointState(  # type: ignore
                        bodyUniqueId=self._body_id,
                        jointIndex=self._joint_idx_dict[joint_name],
                        targetValue=self.cfg.joint_default_values[joint_name],
                        targetVelocity=0.0
                        )
            else:
                assert len(joint_cfg) == len(self._joint_idx_dict)
                for k, joint_name in enumerate(self._joint_idx_dict.keys()):
                    joint_state = joint_cfg[k]
                    self._bc.resetJointState(  # type: ignore
                        bodyUniqueId=self._body_id,
                        jointIndex=self._joint_idx_dict[joint_name], 
                        targetValue=joint_state,
                        targetVelocity=0.0
                        )
        else:
            LOGGER.error(self._CONNECTION_ERROR_MSG)
            raise RuntimeError("Not connect to PyBullet client.")

    def update(self) -> None:
        """ Update physical pybullet state """
        if self.is_connected:
            self.state = self._bc.getJointStates(  # type: ignore
                bodyUniqueId=self._body_id,
                jointIndices=[idx for idx in self._joint_idx_dict.values()]
            )
            self.tcp_link.update()
            self.base_link.update()
            if self._fts: self._fts.update()
        else:
            LOGGER.error(self._CONNECTION_ERROR_MSG)
            raise RuntimeError("Not connect to PyBullet client.")

    @property
    def joint_pos(self) -> tuple[float, ...]:
        state_idx = BulletJointState.JOINT_POSITION
        pos = tuple(joint[state_idx] for joint in self.state)
        return pos

    @property
    def joint_vel(self) -> tuple[float, ...]:
        state_idx = BulletJointState.JOINT_VELOCITY
        vel = tuple(joint[state_idx] for joint in self.state)
        return vel
    
    @property
    def X_arm2plug(self) -> Pose:
        if self.is_connected:
            # Get arm pose
            X_world2arm = self._base.get_X_world2link()  # type: ignore
            # Get tcp pose
            X_world2tcp = self._tcp.get_X_world2link()  # type: ignore
            # Get tcp pose wrt arm pose
            X_arm2tcp = X_world2arm.inverse() * X_world2tcp
        else:
            LOGGER.error(self._CONNECTION_ERROR_MSG)
            X_arm2tcp = Pose()
        return X_arm2tcp

    @property
    def p_arm2plug(self) -> Vector3d:
        return self.X_arm2plug.p

    @property
    def q_arm2plug(self) -> Quaternion:
        return self.X_arm2plug.q

    @property
    def X_world2arm(self) -> Pose:
        if self.is_connected:
            # Get base pose
            X_world2base = self._base.get_X_world2link()  # type: ignore
        else:
            X_world2base = self.cfg.X_world2arm
        return X_world2base
    
    @property
    def p_world2arm(self) -> Vector3d:
        return self.X_world2arm.p
    
    @property
    def q_world2arm(self) -> Quaternion:
        return self.X_world2arm.q
