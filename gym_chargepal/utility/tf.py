""" Helper classes for spatial representations. """
# global
import math
import numpy as np
import quaternionic as quat

# mypy
from numpy import typing as npt
from typing import List, Tuple


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
    def __init__(
        self, 
        q0: float=1.0, 
        q1: float=0.0, 
        q2: float=0.0, 
        q3: float=0.0, 
        order: str='wxyz'
        ) -> None:
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

    def to_3pt_set(self, dist: float=1.0, axes: str='xy') -> npt.NDArray[np.float_]:
        """
        Convert the pose to set of 3 points, the idea being that, 3 (non-colinear) points
        can encode both position and orientation.
        :param dist: Distance from original pose to other two points [m].
        :param axes: Axes of pose orientation to transform other 2 points along. Must be a 
                     two letter combination of x, y and z, in any order.
        :return: A 3x3 np array of each point, with each row containing a single point. The 
                 first point is the original point from the 'pose' argument.
        """
        assert len(axes) == 2, "param axes must be xy, yx, xz, zx, yz, or zy"
        points: List[List[float]] = []
        # First point is just the origin of the pose
        points.append(list(self.pos.as_tuple()))
        axes = axes.replace('x', '0')
        axes = axes.replace('y', '1')
        axes = axes.replace('z', '2')
        tau1 = self.pos.as_array()
        tau2 = 3 * [0.0]
        tau3 = 3 * [0.0]
        tau2[int(axes[0])] = dist
        tau3[int(axes[1])] = dist
        # Get rotation quaternion
        q = self.ori.as_quaternionic()
        # Transform in point 2 and 3 (first rotate then shift)
        pt2 = q.rotate(tau2) + tau1
        points.append(list(pt2))
        pt3 = q.rotate(tau3) + tau1
        points.append(list(pt3))
        return np.array(points, dtype=np.float32)


class Twist:
    """ Representation class of a spatial movement """
    def __init__(
        self,
        vx: float=0.0,
        vy: float=0.0,
        vz: float=0.0,
        wx: float=0.0,
        wy: float=0.0,
        wz: float=0.0, 
        ) -> None:
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.wx = wx
        self.wy = wy
        self.wz = wz

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.vx, self.vy, self.vz, self.wx, self.wy, self.wz)

    def as_array(self) -> npt.NDArray[np.float_]:
        return np.array(self.as_tuple(), dtype=np.float32)


class Wrench:
    """ Class to represent a 6D cartesian wrench """
    def __init__(
        self, 
        fx: float=0.0, 
        fy: float=0.0, 
        fz: float=0.0, 
        mx: float=0.0, 
        my: float=0.0, 
        mz: float=0.0
        ) -> None:
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.fx, self.fy, self.fz, self.mx, self.my, self.mz)

    def as_array(self) -> npt.NDArray[np.float_]:
        return np.array(self.as_tuple(), dtype=np.float32)


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
