from enum import Enum
from math import atan2, degrees
from typing import Tuple

import numpy as np


class Direction(Enum):
    E = 0
    NE = 1
    N = 2
    NW = 3
    W = 4
    SW = 5
    S = 6
    SE = 7
    NONE = 8

    @classmethod
    def get(cls, point, relative_to=None):
        if relative_to is None:
            relative_to = np.zeros_like(point)
        if np.array_equal(point, relative_to):
            return cls.NONE
        diff = point - relative_to
        angle = degrees(atan2(diff[1], diff[0]))
        # There HAS to be a better way... Oh well:
        if -22.5 <= angle < 22.5:
            return cls.E
        elif 22.5 <= angle < 67.5:
            return cls.NE
        elif 67.5 <= angle < 112.5:
            return cls.N
        elif 112.5 <= angle < 157.5:
            return cls.NW
        elif abs(angle) > 157.5:
            return cls.W
        elif -157.5 <= angle < -112.5:
            return cls.SW
        elif -112.5 <= angle < -67.5:
            return cls.S
        elif -67.5 <= angle < -22.5:
            return cls.SE


def npinhole_projection(azimuth, elevation):
    X = 256 * (azimuth / 90) + 128
    Y = 256 * (elevation / 90) + 128
    # X = round(["x_res"] * (1 - 2 * azimuth / settings["x_angle"]) / 2)
    # Y = round(["y_res"] * (1 - 1.8 * elevation / settings["y_angle"]) / 2)
    return int(round(X)), int(round(Y))


def pinhole_projection(azimuth: float, elevation: float) -> Tuple[int, int]:
    X = round(-5.650462986 * azimuth + 130.5760642)
    Y = round(-5.650462986 * elevation + 130.5760642)
    return X, Y


def ninv_pinhole_projection(X: int, Y: int) -> Tuple[float, float]:
    azimuth = (X - 128) * 90 / 256
    elevation = (Y - 128) * 90 / 256
    return azimuth, elevation


def inv_pinhole_projection(X: int, Y: int) -> Tuple[float, float]:
    azimuth = (X - 130.5760642) / -5.650462986
    elevation = (Y - 130.5760642) / -5.650462986
    return azimuth, elevation


def sign(x):
    return x // abs(x)


def sph2cart(r, az, el):
    x = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    y = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    z = r * np.sin(np.radians(el))
    return x, y, z


def yaw_rotation(vec, angle):
    s, c = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    yaw_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    return yaw_matrix @ vec


def pitch_rotation(vec, angle):
    s, c = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    pitch_matrix = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return vec @ pitch_matrix


def line(start, end):
    def bresenham(start, end):
        dx, dy = abs(end[0] - start[0]), -abs(end[1] - start[1])
        err = dx + dy
        x, y = start
        while x != end[0] or y != end[1]:
            yield (x, y)
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sign_x
            if e2 <= dx:
                err += dx
                y += sign_y
        yield (end)

    def sign(x, y):
        """sign(x,y) = -1 iff x<=y; 1 otherwise"""
        return 2 * int(x < y) - 1

    sign_x = sign(start[0], end[0])
    sign_y = sign(start[1], end[1])
    if start[0] == end[0]:
        return ((start[0], i)
                for i in range(start[1], end[1] + sign_y, sign_y))
    else:
        if start[1] == end[1]:
            return ((i, start[1])
                    for i in range(start[0], end[0] + sign_x, sign_x))
        else:
            return bresenham(start, end)


def find_in_matrix(mat, start, end, condition):
    """
    Returns the first element of `mat` that satisfies `condition`.
    `condition` must return a bool.
    """
    for X, Y in line(start, end):
        if condition(mat[Y, X]):
            return np.array((X, Y)), mat[Y, X]
    return None, None


def find_in_matrix_qualitative(mat, start, direction, condition):
    """
    Returns the first element of `mat` that satisfies `condition`.
    The matrix is scanned according to `direction`.
    `condition` must return a bool.
    Throws an IndexError when no element matches the given condition in the
    given direction.
    """
    step = {
        Direction.N: (0, -1),
        Direction.NW: (-1, -1),
        Direction.W: (-1, 0),
        Direction.SW: (-1, 1),
        Direction.S: (0, 1),
        Direction.SE: (1, 1),
        Direction.E: (0, 1),
        Direction.NE: (1, -1),
    }[direction]
    while not condition(mat[start[1], start[0]]):
        start += step
    return None, None
