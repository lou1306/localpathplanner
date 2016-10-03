from enum import Enum
from typing import Tuple, List
from math import degrees, asin, atan2

import numpy as np
import vrep
import cv2


def log_and_retry(func):
    def func_wrapper(self, *args):
        while True:
            try:
                return func(self, *args)
            except ConnectionError as e:
                print("Error in ", func, ": ", e, sep="")
                continue

    return func_wrapper


class VRepError(Enum):
    NOVALUE = 1
    TIMEOUT = 2
    ILLEGAL_OPMODE = 4
    SERVER_ERROR = 8
    SPLIT_PROGRESS = 16
    LOCAL_ERROR = 32
    INIT_ERROR = 64


class VRepObject():
    """
    Simple wrapper around the V-Rep Remote API
    """
    BLOCK = vrep.simx_opmode_blocking

    def __init__(self, client_id: int, name: str):
        self.client_id = client_id
        self.name = name
        ret, handle = vrep.simxGetObjectHandle(client_id, name, self.BLOCK)
        if ret == 0:
            self.handle = handle
        else:
            raise ConnectionError(self._check_errors(ret))

    @staticmethod
    def _check_errors(return_value: int) -> Tuple:
        """
        Returns all the errors associated with a return value.
        """
        if return_value == 0:
            return tuple()
        else:
            return tuple(
                err for err in VRepError
                if bool(return_value & err.value))

    @log_and_retry
    def duplicate(self):
        ret, handles = vrep.simxCopyPasteObjects(self.client_id, [self.handle], self.BLOCK)
        if ret == 0:
            return handles
        else:
            raise ConnectionError(self._check_errors(ret))

    @log_and_retry
    def get_position(self, other: "VRepObject" = None):
        """Retrieve the object position.

        If `handle` is -1, get the absolute position. Otherwise get the position
        relative to the object with the given handle.
        """
        handle = -1
        if other:
            handle = other.handle

        ret, pos = vrep.simxGetObjectPosition(self.client_id, self.handle, handle, self.BLOCK)
        if ret == 0:
            return np.array(pos, np.float32)
        else:
            raise ConnectionError(self._check_errors(ret))

    @log_and_retry
    def get_velocity(self) -> Tuple:
        ret, linear, angular = vrep.simxGetObjectVelocity(self.client_id, self.handle, self.BLOCK)
        if ret == 0:
            return np.array(linear, np.float32), np.array(angular, np.float32)
        else:
            raise ConnectionError(self._check_errors(ret))

    @log_and_retry
    def get_spherical(self, other: "VRepObject" = None, offset: object = [0, 0, 0]) -> object:
        """Spherical coordinates of object.

        Azimuth is CCW from X axis.
        0       Front
        90      Leftside
        +/-180  Back
        -90     Rightside

        Elevation is respective to the XY plane.
        0       Horizon
        90      Zenith
        -90     Nadir
        """
        while True:
            try:
                pos = self.get_position(other)
                pos += offset
                dist = np.linalg.norm(pos)
                azimuth = degrees(atan2(pos[1], pos[0]))
                elevation = degrees(asin(pos[2] / dist))
                return dist, azimuth, elevation
            except ConnectionError:
                continue

    @log_and_retry
    def get_orientation(self, other: "VRepObject" = None):
        """Retrieve the object orientation (as Euler angles)
        """
        handle = -1
        if other:
            handle = other.handle

        ret, euler = vrep.simxGetObjectOrientation(
            self.client_id, self.handle, handle, self.BLOCK)
        if ret == 0:
            return np.array(euler, np.float32)
        else:
            raise ConnectionError(self._check_errors(ret))

    @log_and_retry
    def set_position(self, pos, other: "VRepObject" = None):
        """Sets the position.

        pos: 3-valued list or np.array (x,y,z coordinates in meters)
        """
        handle = -1
        if other:
            handle = other.handle

        ret = vrep.simxSetObjectPosition(self.client_id, self.handle, handle, pos, self.BLOCK)
        if ret != 0:
            raise ConnectionError(self._check_errors(ret))

    @log_and_retry
    def set_orientation(self, euler: Tuple[float, float, float]):
        """
        Sets the absolute orientation of the object
        """
        ret = vrep.simxSetObjectOrientation(self.client_id, self.handle, -1, euler, self.BLOCK)
        if ret != 0:
            raise ConnectionError(self._check_errors(ret))


class VRepDepthSensor(VRepObject):
    @log_and_retry
    def get_depth_buffer(self) -> np.ndarray:
        ret, res, d = vrep.simxGetVisionSensorDepthBuffer(self.client_id, self.handle, self.BLOCK)
        if ret != 0:
            raise ConnectionError(self._check_errors(ret))
        else:
            d = np.array(d, np.float32).reshape((res[1], res[0]))
            d = np.flipud(d)  # the depth buffer is upside-down
            d = cv2.resize(d, (256, 256))  # TODO make codebase resolution-agnostic
            return res, d


class VRepDummy(VRepObject):
    pass


class VRepClient():
    def __init__(self, host: str, port: int):
        self._conn_id = vrep.simxStart(host, port, True, True, -100, 5)
        if self._conn_id == -1:
            raise ConnectionError("Connection to {}:{} failed".format(host, port))

    def get_object(self, name: str) -> "VRepObject":
        return VRepObject(self._conn_id, name)

    def get_depth_sensor(self, name: str) -> VRepDepthSensor:
        return VRepDepthSensor(self._conn_id, name)

    # TODO wrap return codes in exceptions
    def create_dummy(self, pos: List[float], size: float = 0.2):
        ret, dummy_handle = vrep.simxCreateDummy(self._conn_id, size, None, vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self._conn_id, dummy_handle, -1, pos, vrep.simx_opmode_blocking)
        # return VRepDummy(self._conn_id, dummy_handle)

    def close_connection(self):
        vrep.simxFinish(self._conn_id)
