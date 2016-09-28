from enum import Enum
from typing import Tuple
from math import degrees, asin, atan2

import numpy as np
import vrep

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


    def get_position(self, handle: int=-1):
        """Retrieve the object position.

        If `handle` is -1, get the absolute position. Otherwise get the position
        relative to the object with the given handle.
        """
        ret, pos = vrep.simxGetObjectPosition(
            self.client_id, self.handle, handle, self.BLOCK)
        if ret == 0:
            return np.array(pos, np.float32)
        else:
            raise ConnectionError(self._check_errors(ret))

    def get_spherical(self, handle=-1, offset = [0,0,0]):
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
        pos = self.get_position(handle)
        pos += offset
        dist = np.linalg.norm(pos)
        azimuth = degrees(atan2(pos[1],pos[0]))
        elevation = degrees(asin(pos[2]/dist))
        return dist, azimuth, elevation

    
    def get_orientation(self, handle=-1):
        """Retrieve the object orientation (as Euler angles)
        """
        ret, euler = vrep.simxGetObjectOrientation(
            self.client_id, self.handle, handle, self.BLOCK)
        if ret == 0:
            return np.array(euler, np.float32)
        else:
            raise ConnectionError(self._check_errors(ret))

    def set_position(self, pos, handle=-1):
        """Sets the position.

        pos: 3-valued list or np.array (x,y,z coordinates in meters)
        """
        ret = vrep.simxSetObjectPosition(self.client_id, self.handle, handle, pos, self.BLOCK)
        if ret != 0:
            raise ConnectionError(self._check_errors(ret))

    def set_orientation(self, euler):
        """
        Sets the absolute orientation
        """
        ret = vrep.simxSetObjectOrientation(self.client_id, self.handle, -1, euler, self.BLOCK)
        if ret != 0:
            raise ConnectionError(self._check_errors(ret))

class VRepDepthSensor(VRepObject):
    def get_depth_buffer(self) -> Tuple:
        ret, res, d = vrep.simxGetVisionSensorDepthBuffer(self.client_id, self.handle, self.BLOCK)
        if ret != 0:
            raise ConnectionError(self._check_errors(ret))
        else:
            return res, d

class VRepDummy(VRepObject):
    pass

class VRepConnection():
    def __init__(self, host:str, port:int):
        self._conn_id = vrep.simxStart(host, port, True, True, -100, 5)
        if self._conn_id == -1:
            raise ConnectionError("Connection to {}:{} failed".format(host, port))
    def get_object(self, name: str):
        return VRepObject(self._conn_id, name)
    def get_depth_sensor(self, name: str):
        return VRepDepthSensor(self._conn_id, name)

    def create_dummy(self, pos, size: float=0.2):
        ret, dummy_handle = vrep.simxCreateDummy(self._conn_id, size, None, vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self._conn_id, dummy_handle, -1, pos, vrep.simx_opmode_blocking)
        #return VRepDummy(self._conn_id, dummy_handle)

    def close_connection(self):
        vrep.simxFinish(self._conn_id)