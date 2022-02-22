import numpy as np
# import quaternion
import quaternionic

# MyPy
from typing import Tuple


def wrap(x: float, m: float, m_: float) -> float:
    """ Make sure the range is between m and M.
    :param x: a scalar
    :param m: minimum possible value in range
    :param m_: maximum possible value in range
    Wraps 'x' so m <= x <= M; but unlike 'bound()' which truncates,
    'wrap()' wraps x around the coordinate system defined by m,M.
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = m_ - m

    while x > m_:
        x = x - diff

    while x < m:
        x = x + diff

    return x


def bullet2numpy_repr(q_bullet: Tuple[float, ...]) -> np.quaternion:
    """ Change quaternion representation.
    PyBullet [w, x, y, z]
    Numpy Quaternion [x, y, z, w]
    """
    return quaternionic.array([q_bullet[3], q_bullet[0], q_bullet[1], q_bullet[2]])
    # return np.quaternion(q_bullet[3], q_bullet[0], q_bullet[1], q_bullet[2])
