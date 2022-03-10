import numpy as np

# MyPy
from typing import Tuple
from gym_chargepal.sensors.sensor_target_adpstd import TargetSensor


class CurriculumPegInHole(object):
    """ Curriculum class. To schedule the task difficulty over time. """
    def __init__(self, tgt_sensor: TargetSensor):
        self._level = 0
        self._ts = tgt_sensor
        self._curriculum = [self._level0, self._level1, self._level2, self._level3, self._level4, self._level5]

    def sample(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        :return: p0: start position
                 q0: start orientation
        """
        p0, q0 = self._curriculum[self._level]()
        return p0, q0

    def _level0(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        p0 = np.array(self._ts.get_pos(), dtype=np.float32)
        p0[1] -= 0.04 + 0.01 * np.random.randn(1)
        return tuple(p0), self._ts.get_ori()

    def _level1(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        p0 = np.array(self._ts.get_pos(), dtype=np.float32)
        p0[1] -= 0.05 + 0.01 * np.random.randn(1)
        return tuple(p0), self._ts.get_ori()

    def _level2(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        p0 = np.array(self._ts.get_pos(), dtype=np.float32)
        p0[1] -= 0.06 + 0.01 * np.random.randn(1)
        return tuple(p0), self._ts.get_ori()

    def _level3(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        p0 = np.array(self._ts.get_pos(), dtype=np.float32)
        p0[1] -= 0.07 + 0.01 * np.random.randn(1)
        return tuple(p0), self._ts.get_ori()

    def _level4(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        p0 = np.array(self._ts.get_pos(), dtype=np.float32)
        p0[1] -= 0.08 + 0.01 * np.random.randn(1)
        return tuple(p0), self._ts.get_ori()

    def _level5(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        p0 = np.array(self._ts.get_pos(), dtype=np.float32)
        p0 = p0 + np.array((0.0, -0.1, 0.0)) + np.array((0.01, 0.001, 0.01)) * np.random.randn(3)
        return tuple(p0), self._ts.get_ori()

    def level_up(self) -> None:
        self._level += 1
        if self._level > len(self._curriculum) - 1:
            self._level = len(self._curriculum) - 1

    def level_down(self) -> None:
        self._level -= 1
        if self._level < 0:
            self._level = 0

    def get_level(self) -> int:
        return self._level
