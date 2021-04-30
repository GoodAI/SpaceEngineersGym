from typing import Union

import numpy as np


class Point3D(object):
    """Wrapper around numpy array to represent a 3D point."""

    def __init__(self, point: np.ndarray):
        self.x, self.y, self.z = point

    def tolist(self):
        return self.toarray().tolist()

    def toarray(self):
        return self.__array__()

    def __copy__(self):
        array_result = self.toarray()
        return Point3D(array_result)

    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"

    def __add__(self, other) -> Union[np.ndarray, "Point3D"]:
        array_result = other + self.toarray()
        if isinstance(other, Point3D):
            return Point3D(array_result)
        return array_result

    def __sub__(self, other) -> Union[np.ndarray, "Point3D"]:
        array_result = other - self.toarray()
        if isinstance(other, Point3D):
            return Point3D(array_result)
        return array_result

    def __neg__(self):
        return Point3D(-self.toarray())

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype:
            return np.array([self.x, self.y, self.z], dtype=dtype)
        else:
            return np.array([self.x, self.y, self.z])


def in_absolute_frame(
    pos_relative: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: Point3D,
) -> np.ndarray:
    pos_abs = pos_relative @ np.linalg.inv(rotation_matrix)
    pos_abs -= translation
    return pos_abs


def in_relative_frame(
    pos_abs: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: Point3D,
) -> np.ndarray:
    """
    Inverse transform of `in_absolute_frame`.
    """
    pos_relative = pos_abs + translation
    pos_relative = pos_relative @ rotation_matrix
    return pos_relative


def normalize_angle(angle: float) -> float:
    """
    :param angle: angle in radian
    :return: angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2 * np.pi

    while angle < -np.pi:
        angle += 2 * np.pi

    return angle
