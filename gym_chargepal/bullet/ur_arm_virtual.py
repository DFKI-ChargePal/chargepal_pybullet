""" This file defines the Virtual UR arm class """
from __future__ import annotations

# global
import logging
import numpy as np
import pybullet as p
from dataclasses import dataclass
from rigmopy import Quaternion, Vector3d
from pybullet_utils.bullet_client import BulletClient

# local
from gym_chargepal.worlds.world import World
from gym_chargepal.utility.cfg_handler import ConfigHandler

# typing
from typing import Any
from numpy import typing as npt


LOGGER = logging.getLogger(__name__)


@dataclass
class VirtualURArmCfg(ConfigHandler):
    mass_min: float = 1e-3
    inertia_min: float = 1e-6
    mass_generic: float = 1.0
    inertia_generic: float = 1.0


class VirtualURArm:

    def __init__(self, config: dict[str, Any], world: World) -> None:
        # Create configuration and overwrite values
        self.cfg = VirtualURArmCfg()
        self.cfg.update(**config)
        # PyBullet elements
        self._bc: BulletClient | None = None
        self.robot_id = -1
        # Save references
        self.world = world

    @property
    def bullet_client(self) -> BulletClient:
        if self._bc is not None:
            return self._bc
        else:
            raise RuntimeError(f"No connection to PyBullet. Please fist connect via {__name__}.connect()")

    @property
    def is_connected(self) -> bool:
        return True if self._bc else False

    def connect(self) -> None:
        # Initialize PyBullet simulation
        self._bc = BulletClient(connection_mode=p.DIRECT)
        self._bc.setRealTimeSimulation(False)
        self._bc.resetSimulation()
        self._bc.setPhysicalEngineParameter(deterministicOverlappingPairs=1)
        # Load and manipulate robot description
        self._load_robot()
        self._model_homogenization()

    def disconnect(self) -> None:
        if self._bc is not None:
            connection_info = self._bc.getConnectionInfo()
            if connection_info['isConnected'] > 0:
                self._bc.disconnect()
            self._bc = None
        self.robot_id = -1

    def calc_inertial_matrix(self, joint_pos: tuple[float, ...]) -> npt.NDArray[np.float64]:
        mm = self.bullet_client.calculateMassMatrix(
            bodyUniqueId=self.robot_id,
            objPositions=joint_pos
        )
        dim = len(joint_pos)
        return np.array(mm, dtype=np.float64).reshape([dim, dim])

    def _load_robot(self) -> None:
        arm_urdf_path = self.world.urdf_pkg_path.joinpath(self.world.cfg.robot_urdf)
        # Initial pose
        p0 = Vector3d()
        q0 = Quaternion().from_axis_angle([0.0, 0.0, np.pi])  # Rotate arm base by 180 degrees around Z axes.
        # Load model
        self.robot_id = self.bullet_client.loadURDF(
            fileName=str(arm_urdf_path),
            basePosition=p0.xyz,
            baseOrientation=q0.xyzw
        )
        # Set gravity
        self.bullet_client.setGravity(*self.world.cfg.gravity)

    def _model_homogenization(self) -> None:
        """ Change the model mass and inertias to homogenize the  inertia matrix
        See https://arxiv.org/pdf/1908.06252.pdf for a motivation for this setting.
        """
        # Set all masses and inertias to minimal (but stable) values
        ip_min_diag = 3 * [self.cfg.inertia_min]
        n_links = self.bullet_client.getNumJoints(bodyUniqueId=self.robot_id)
        for idx in range(n_links):
            self.bullet_client.changeDynamics(
                bodyUniqueId=self.robot_id,
                linkIndex=idx,
                mass=self.cfg.mass_min,
                localInertiaDiagonal=ip_min_diag
            )
        # Only give the tcp segment a generic mass and inertia
        ip_generic_diag = 3 * [self.cfg.inertia_generic]
        self.bullet_client.changeDynamics(
            bodyUniqueId=self.robot_id,
            linkIndex=self.world.ur_arm.tcp_link.idx,
            mass=self.cfg.mass_generic,
            localInertiaDiagonal=ip_generic_diag
            )
