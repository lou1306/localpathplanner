from __future__ import division, print_function

from enum import Enum
from math import radians
from time import sleep
#from urllib.request import urlopen
import json


import cv2
import numpy as np


from functions import pinhole_projection
from pid import PID
from vrep_object import VRepClient, VRepObject


def radius(dist):
    return max(int(120 // dist), 1)


class Visibility(Enum):
    VISIBLE = 1
    NOT_VISIBLE = 2
    UNREACHABLE = 3


class Drone(VRepObject):
    MAX_ANGLE = 45  # degrees
    MAX_DEPTH = 10  # meters
    RADIUS = 0.5  # meters

    def __init__(self, client):
        self._body = client.get_object("Quadricopter_base")
        super(Drone, self).__init__(client.id, self._body.handle, "")
        self._target = client.get_object("Quadricopter_target")
        self._sensor = client.get_object("fast3DLaserScanner_sensor")

        self._rotation_pid = PID(0.2, 0.05, 0.2, 1, max_int=3)
        self._altitude_pid = PID(0.2, 0.02, 0.2, 1)
        self._pid = PID(4.5, 0.01, 0.1, 3, 0.15, max_int=10)

        self.total_distance = 0
        self.sensor_offset = - self._body.get_position(self._sensor)

    def altitude_adjust(self, goal):
        good = err = 0.5  # meters
        while abs(err) >= good:
            goal_pos = goal.get_position(self._body)
            err = goal_pos[2]  # z-coordinate
            correction = self._altitude_pid.control(err)
            if __debug__:
                print("Adjusting altitude...", correction)
            self._target.set_position(self._target.get_position() + np.array([0, 0, correction]))
            self.total_distance += np.linalg.norm(correction)
            sleep(1)
        else:
            if __debug__:
                print("...Adjusted. Goal at {} m".format(err))
            self._altitude_pid.reset()
            sleep(2)  # Wait for the drone to stabilize

    def can_reach(self, goal):
        dist, azimuth, elevation = goal.get_spherical(self._body, self.sensor_offset)
        delta = goal.get_position(self._target)
        h_dist = np.linalg.norm(delta[0:2])

        res, d = self._sensor.get_depth_buffer()

        X, Y = pinhole_projection(azimuth, elevation)
        ball_r = radius(dist)
        mask = cv2.circle(np.zeros_like(d), (X, Y), ball_r, 1, -1)
        try:
            min_depth = np.min(d[mask == 1]) * self.MAX_DEPTH
        except ValueError:
            # Mask has no white pixel.
            self.lock(goal)
            return self.can_reach(goal)
        return h_dist < 1 or dist - min_depth < -0.5 or min_depth == self.MAX_DEPTH, d, min_depth, mask

    def reset_controllers(self):
        self._pid.reset()
        self._altitude_pid.reset()
        self._rotation_pid.reset()

    def rotate_towards(self, goal):
        """Rotates the drone until it points towards the goal.

        Actually, the function rotates the `target` object which is then followed
        by the `drone` (inside V-REP).

        `sensor_offset`: position of the sensor relative to the drone. Needed for a
        better azimuth value.
        """
        good = azimuth = 5  # Degrees
        while abs(azimuth) >= good:
            euler = self._target.get_orientation()
            __, azimuth, __ = goal.get_spherical(self._body, self.sensor_offset)
            correction_angle = self._rotation_pid.control(azimuth)
            if __debug__:
                print("Adjusting orientation...", correction_angle)
            euler[2] += radians(correction_angle)  # euler[2] = Yaw
            self._target.set_orientation(euler)
            sleep(1)
        else:
            if __debug__:
                print("...Adjusted. Goal at {} degrees".format(azimuth))
            self._rotation_pid.reset()
            self.stabilize()  # Wait for the drone to stabilize on the new angle

    def lock(self, goal):
        __, azimuth, elevation = goal.get_spherical(self._body, self.sensor_offset)
        X, Y = pinhole_projection(azimuth, elevation)
        if abs(elevation) > self.MAX_ANGLE or not 0 <= Y < 256:
            self.altitude_adjust(goal)
        self.rotate_towards(goal)

    def stabilize(self):
        eps = 0.001
        if __debug__:
            print("UAV stabilization in progress...")
        while True:
            lin_v, ang_v = self._body.get_velocity()
            if all(i < eps for i in lin_v) and all(i < eps for i in ang_v):
                if __debug__:
                    sleep(0.5)
                    print("...done.")
                return
            else:
                sleep(0.05)

    def step_towards(self, goal):
        target_pos = self._target.get_position()
        correction = self._pid.control(-self._target.get_position(goal))
        self.total_distance += np.linalg.norm(correction)
        self._target.set_position(target_pos + correction)

    def escape(self, goal):
        self.rotate(60)
        __, d = self._sensor.get_depth_buffer()
        left_space = len(d[d == 1])
        self.rotate(-120)
        __, d = self._sensor.get_depth_buffer()
        right_space = len(d[d == 1])
        go_left = left_space >= right_space

    def rotate(self, angle):
        self._rotation_pid.reset()
        while abs(angle) > 2:
            euler = self._target.get_orientation()
            correction = self._rotation_pid.control(angle)
            angle -= correction
            euler[2] += radians(correction)  # euler[2] = Yaw
            self._target.set_orientation(euler)
            sleep(1)
        self.stabilize()


class MPDrone(Drone):
    URL = "http://127.0.0.1:56781/{}"
    def __init__(self):
        pass


    def _get_mavlink(self):
        with urlopen(self.URL.format("mavlink/")) as req:
            return json.loads(req.read().decode('utf-8'))

    def get_position(self, other = None):
        mavlink = self._get_mavlink()
        lat = mavlink["GPS_RAW_INT"]["msg"]["lat"] / 10**7
        lon = mavlink["GPS_RAW_INT"]["msg"]["lon"] / 10**7
        alt = mavlink["GPS_RAW_INT"]["msg"]["alt"] / 10**3
        return lat, lon, alt

    def rotate_towards(self, goal):
        lat, lon, alt = goal
        string = "guided?lat={}&lon={}&alt={}".format(lat,lon,alt)
        with urlopen(self.URL.format(string)) as req:
            print(req.read().decode('utf-8'))




