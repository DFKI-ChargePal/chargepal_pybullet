""" Helper classes for spatial representations. """
# global
import math
import numpy as np
import quaternionic as quat

# mypy
from numpy import typing as npt
from typing import Tuple


class Translation:
    """ Translation class """
    def __init__(self, x: float=0.0, y: float=0.0, z: float=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.x, self.y, self.z)

    def as_array(self) -> npt.NDArray[np.float_]:
        return np.array(self.as_tuple(), dtype=np.float32)


class Quaternion:
    """ Quaternion class """
    def __init__(self, q0: float=1.0, q1: float=0.0, q2: float=0.0, q3: float=0.0, order: str='wxyz') -> None:
        assert len(order) == 4
        idx_map = {io: idx for io, idx in zip(order, range(4))}
        in_quat = [q0, q1, q2, q3]
        self.w = in_quat[idx_map['w']]
        self.x = in_quat[idx_map['x']]
        self.y = in_quat[idx_map['y']]
        self.z = in_quat[idx_map['z']]

    def as_tuple(self, order: str='wxyz') -> Tuple[float, ...]:
        q = [1.0, 0.0, 0.0, 0.0]
        for idx, io in enumerate(order):
            q[idx] = self.__getattribute__(io)
        return tuple(q)

    def as_array(self, order: str='wxyz') -> npt.NDArray[np.float_]:
        return np.array(self.as_tuple(order=order), dtype=np.float32)

    def as_quaternionic(self) -> quat.QuaternionicArray:
        return quat.array([self.w, self.x, self.y, self.z])

    def from_euler_angles(self, roll: float, pitch: float, yaw: float) -> None:
        """ Fill the quaternion from euler angels. 
        Convention is 3-2-1: 
            1) Rotate around yaw-axis
            2) Rotate around pitch-axis
            3) Rotate around roll-axis
        """
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        self.w = cr * cp * cy + sr * sp * sy;
        self.x = sr * cp * cy - cr * sp * sy;
        self.y = cr * sp * cy + sr * cp * sy;
        self.z = cr * cp * sy - sr * sp * cy;


class Euler:
    """ Euler angle representation class """
    def __init__(self, x: float=0.0, y: float=0.0, z: float=0.0) -> None:
        self.roll = x
        self.pitch = y
        self.yaw = z

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.roll, self.pitch, self.yaw)

    def as_array(self) -> npt.NDArray[np.float_]:
        return np.array(self.as_tuple(), dtype=np.float32)


class Pose:
    """ Spatial pose class """
    def __init__(self, pos: Translation, ori: Quaternion) -> None:
        self.pos = pos
        self.ori = ori

    def as_tuple(self, q_order: str='wxyz') -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        return self.pos.as_tuple(), self.ori.as_tuple(order=q_order)

    def as_array(self, q_order: str='wxyz') -> Tuple[npt.NDArray[np.float_], ...]:
        return self.pos.as_array(), self.ori.as_array(order=q_order)

    def as_flat_array(self, q_order: str='wxyz') -> npt.NDArray[np.float_]:
        return np.array(self.pos.as_tuple() + self.ori.as_tuple(order=q_order), dtype=np.float32)


class RndPoseGenerator:
    """ Random pose generator class """
    def __init__(self, lin_var: Tuple[float, ...], eul_var: Tuple[float, ...]) -> None:
        assert len(lin_var) == 3
        assert len(eul_var) == 3
        self.lin_var = Translation(*lin_var)
        self.eul_var = Euler(*eul_var)

    def rand_quat(self, order: str='wxyz') -> npt.NDArray[np.float_]:
        rand_eul: Tuple[float, ...] = tuple(self.eul_var.as_array() * np.random.randn(3))
        rand_quat = Quaternion()
        rand_quat.from_euler_angles(*rand_eul)
        return rand_quat.as_array(order=order)

    def rand_linear(self) -> npt.NDArray[np.float_]:
        return self.lin_var.as_array() * np.random.randn(3)
