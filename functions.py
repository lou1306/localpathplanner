from enum import Enum
from math import atan2, degrees

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
    O = 8
    @classmethod
    def get(cls, point, relative_to=None):
        if relative_to is None:
            relative_to = np.zeros_like(point)
        if np.array_equal(point, relative_to):
            return cls.O
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

def old_pinhole_projection(azimuth, elevation, settings):
    X = round(settings["x_res"] * (1 -2*azimuth/settings["x_angle"])/2)
    #Y = res[1] // 2 # Assuming Z_goal == Z_target (TODO handle )
    Y = round(settings["y_res"] * (1 - 1.8*elevation/settings["y_angle"])/2)
    return X, Y

def pinhole_projection(azimuth, elevation, settings):
    X = round(-5.650462986 * azimuth + 130.5760642)
    Y = round(-5.650462986 * elevation + 130.5760642)
    #Y = int(-5.525447299 * elevation + 141.2729276)
    return X, Y

def old_inv_pinhole_projection(X, Y, settings):
    azimuth = settings["x_angle"] * (1- 2*X / settings["x_res"])
    elevation = settings["y_angle"] * (1 - 1.8*Y / settings["y_res"])
    return azimuth, elevation

def inv_pinhole_projection(X, Y, settings):
    azimuth = (X - 130.5760642)/ -5.650462986
    elevation = (Y - 130.5760642)/ -5.650462986
    return azimuth, elevation

def sign(x):
    return x // abs(x)

def sph2cart(r, az, el):
    x = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    y = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    z = r * np.sin(np.radians(el))
    return x,y,z

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
        [-s, 9, c]
    ])
    return vec @ pitch_matrix

