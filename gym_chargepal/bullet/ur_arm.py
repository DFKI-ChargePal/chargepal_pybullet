""" This file defines the UR arm robot class """
from __future__ import annotations

# global
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from rigmopy import Pose, Quaternion, Vector3d, Vector6d
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
from gym_chargepal.utility.cfg_handler import ConfigHandler
from gym_chargepal.bullet.utility import create_joint_index_dict, get_joint_idx

# mypy
from typing import Any, Deque
from numpy import typing as npt


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
    tool_com_link_names: tuple[str, ...] = ('plug_root', 'plug')
    base_link_name: str = 'base'
    ft_enable: bool = False
    ft_buffer_size: int = 10
    ft_link_name: str = 'ft_sensor_wrench'
    ft_joint_name: str = 'mounting_to_wrench'
    ft_range: tuple[float, ...] = (500.0, 500.0, 1200.0, 15.0, 15.0, 12.0)
    ft_overload: tuple[float, ...] = (2000.0, 2000.0, 4000.0, 30.0, 30.0, 30.0)
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
        self._joint_idx_dict: dict[str, int] = {}
        # Arm links and joints
        self._arm_state: tuple[tuple[float], ...] | None = None
        self._base: BodyLink | None = None
        self._tcp: BodyLink | None = None
        self._fts: BodyLink | None = None
        self._tool_com_links: list[BodyLink] = []
        # Tool physics
        self._update_physics = True
        self.tool_mass = 0.0
        self._tool_com: Vector3d | None = None
        # FT sensor
        self.fts_enable = self.cfg.ft_enable
        self._fts_joint_idx: int | None = None
        self._fts_state: tuple[tuple[float], ...] | None = None
        self.fts_mass = 0.0
        self._fts_com: Vector3d | None = None
        self.ft_min = -np.array(self.cfg.ft_range, dtype=np.float64)
        self.ft_max = np.array(self.cfg.ft_range, dtype=np.float64)
        self.sensor_readings: Deque[npt.NDArray[np.float64]] = deque(maxlen=self.cfg.ft_buffer_size)

    @property
    def is_connected(self) -> bool:
        """ Check if object is connect to PyBullet"""
        return True if self._bc else False

    @property
    def bullet_client(self) -> BulletClient:
        if self.is_connected:
            return self._bc
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def bullet_body_id(self) -> int:
        if self.is_connected and self._body_id:
            return self._body_id
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def base_link(self) -> BodyLink:
        if self.is_connected and self._base:
            return self._base
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def tcp_link(self) -> BodyLink:
        if self.is_connected and self._tcp:
            return self._tcp
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def fts_link(self) -> BodyLink:
        if self.is_connected and self._fts:
            return self._fts
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def fts_joint_idx(self) -> int:
        if self._fts_joint_idx:
            return self._fts_joint_idx
        else:
            raise RuntimeError("FT-sensor index is not set yet. Please get it first from PyBullet.")

    def _enable_fts(self) -> None:
        # enable Force-Torque sensor
        if self.is_connected:
            self.bullet_client.enableJointForceTorqueSensor(
                bodyUniqueId=self.bullet_body_id,
                jointIndex=self.fts_joint_idx,
                enableSensor=True
                )
        else:
            LOGGER.warn(f"Enable ft-sensor is not possible. {self._CONNECTION_ERROR_MSG}")

    def _disable_fts(self) -> None:
        # disable Force-Torque sensor
        if self.is_connected:
            self.bullet_client.enableJointForceTorqueSensor(
                bodyUniqueId=self.bullet_body_id,
                jointIndex=self.fts_joint_idx,
                enableSensor=False
            )
        else:
            LOGGER.warn(f"Disable ft-sensor is not possible. {self._CONNECTION_ERROR_MSG}")

    def connect(self, bullet_client: BulletClient, body_id: int) -> None:
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
        for link_name in self.cfg.tool_com_link_names:
            self._tool_com_links.append(
                BodyLink(
                name=link_name,
                bullet_client=bullet_client,
                body_id=body_id
                )
            )
        if self.fts_enable:
            self._fts = BodyLink(name=self.cfg.ft_link_name, bullet_client=bullet_client, body_id=body_id)
            self._fts_joint_idx = get_joint_idx(body_id, self.cfg.ft_joint_name, bullet_client)
            self._enable_fts()
            self._ft_mass = self._fts.mass

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
            self._update_physics = True
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    def update(self) -> None:
        """ Update physical pybullet state """
        if self.is_connected:
            self._arm_state = self.bullet_client.getJointStates(
                bodyUniqueId=self._body_id,
                jointIndices=[idx for idx in self._joint_idx_dict.values()]
            )
            self.tcp_link.update()
            self.base_link.update()
            
            if self.fts_enable:
                self.fts_link.update()
                self._fts_state = self.bullet_client.getJointState(
                    bodyUniqueId=self._body_id, 
                    jointIndex=self.fts_joint_idx
                    )
                if self._update_physics:
                    # Update ft-sensor physics
                    self.fts_mass = self.fts_link.mass
                    q_inertial2arm = self.fts_link.q_link2inertial.inverse() * self.fts_link.q_world2link.inverse() * self.q_world2arm
                    self._fts_com = q_inertial2arm.apply(self.fts_link.p_link2inertial)

            if self._update_physics:
                # Update ft-sensor physics
                self.fts_mass = self.fts_link.mass
                # Update tool physics
                N = len(self._tool_com_links)
                com_x, com_y, com_z = 0.0, 0.0, 0.0
                tool_mass = 0.0
                for tool_link in self._tool_com_links:
                    tool_mass += tool_link.mass
                    q_inertial2arm = tool_link.q_link2inertial.inverse() * tool_link.q_world2link.inverse() * self.q_world2arm
                    p_link2inertial_arm = q_inertial2arm.apply(tool_link.p_link2inertial).xyz
                    com_x += p_link2inertial_arm[0]
                    com_x += p_link2inertial_arm[1]
                    com_x += p_link2inertial_arm[2]
                self.tool_mass = tool_mass
                self._tool_com = Vector3d().from_xyz((com_x/N, com_y/N, com_z/N))
                # Update physics only after reset.
                self._update_physics = False
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def joint_pos(self) -> tuple[float, ...]:
        state_idx = BulletJointState.JOINT_POSITION
        assert self._arm_state
        pos = tuple(joint[state_idx] for joint in self._arm_state)
        return pos

    @property
    def joint_vel(self) -> tuple[float, ...]:
        state_idx = BulletJointState.JOINT_VELOCITY
        assert self._arm_state
        vel = tuple(joint[state_idx] for joint in self._arm_state)
        return vel
    
    @property
    def X_arm2plug(self) -> Pose:
        if self.is_connected:
            # Get arm pose
            X_world2arm = self.base_link.X_world2link
            # Get tcp pose
            X_world2tcp = self.tcp_link.X_world2link
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
            X_world2base = self.base_link.X_world2link
        else:
            X_world2base = self.cfg.X_world2arm
        return X_world2base
    
    @property
    def p_world2arm(self) -> Vector3d:
        return self.X_world2arm.p
    
    @property
    def q_world2arm(self) -> Quaternion:
        return self.X_world2arm.q

    @property
    def wrench(self) -> Vector6d:
        state_idx = BulletJointState.JOINT_REACTION_FORCE
        assert self._fts_state
        wrench: tuple[float, ...] = self._fts_state[state_idx]
        self.sensor_readings.append(np.array(wrench, dtype=np.float64))
        # Get sensor state and bring values in a range between -1.0 and +1.0
        mean_wrench = np.mean(self.sensor_readings, axis=0, dtype=np.float64)
        norm_wrench = Vector6d().from_xyzXYZ(np.clip(mean_wrench, self.ft_min, self.ft_max) / self.ft_max)
        return norm_wrench

    @property
    def X_arm2fts(self) -> Pose:        
        if self.is_connected:
            # Get arm pose
            X_world2arm = self.base_link.X_world2link
            # Get ft-sensor pose
            X_world2fts = self.fts_link.X_world2link
            # Get tcp pose wrt arm pose
            X_arm2fts = X_world2arm.inverse() * X_world2fts
        else:
            LOGGER.error(self._CONNECTION_ERROR_MSG)
            X_arm2fts = Pose()
        return X_arm2fts
    
    @property
    def p_arm2fts(self) -> Vector3d:
        return self.X_arm2fts.p
    
    @property
    def q_arm2fts(self) -> Quaternion:
        return self.X_arm2fts.q
    
    @property
    def tool_com(self) -> Vector3d:
        if self._tool_com:
            return self._tool_com
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)

    @property
    def fts_com(self) -> Vector3d:
        if self._fts_com:
            return self._fts_com
        else:
            raise RuntimeError(self._CONNECTION_ERROR_MSG)


