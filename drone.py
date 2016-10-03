from math import radians
from time import sleep

import numpy as np
import cv2

from pid import PID
from vrep_object import VRepClient, VRepObject

from functions import pinhole_projection


def radius(dist):
    return max(int(150 // dist), 1)

class Drone(VRepObject):
    MAX_ANGLE = 45 # degrees
    MAX_DEPTH = 10 # meters
    RADIUS = 0.5 # meters

    def __init__(self, client: VRepClient):
        self._client = client
        self._target = client.get_object("Quadricopter_target")
        self._body = client.get_object("Quadricopter_base")
        self._sensor = client.get_depth_sensor("fast3DLaserScanner_sensor")

        self._rotation_pid = PID(0.2, 0.05, 0.2, 1, max_int=3)
        self._altitude_pid = PID(0.2, 0.02, 0.2, 1)
        self._pid = PID(4.5, 0.01, 0.1, 3, 0.15, max_int=10)

        self.total_distance = 0
        self.sensor_offset = - self._body.get_position(self._sensor)
        self.handle = self._body.handle

    def altitude_adjust(self, goal: VRepObject) -> object:
        GOOD = err =  0.5  # meters
        while abs(err) >= GOOD:
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

    def can_reach(self, goal: VRepObject):
        dist, azimuth, elevation = goal.get_spherical(self._body, self.sensor_offset)
        #delta = goal.get_position(self._target)
        #h_dist = np.linalg.norm(dist[0:2])
        h_dist = dist * np.cos(elevation)

        res, d = self._sensor.get_depth_buffer()

        X, Y = pinhole_projection(azimuth, elevation)
        ball_r = radius(dist)
        mask = cv2.circle(np.zeros_like(d), (X, Y), ball_r, 1, -1)
        try:
            min_depth = np.min(d[mask == 1]) * self.MAX_DEPTH
        except ValueError:
            # Mask has no white pixel.
            raise ValueError
        return h_dist < 1 or dist - min_depth < -0.5 or min_depth == self.MAX_DEPTH, d, min_depth, mask

    def reset_controllers(self):
        self._pid.reset()
        self._altitude_pid.reset()
        self._rotation_pid.reset()

    def rotate_towards(self, goal: VRepObject):
        """Rotates the drone until it points towards the goal.

        Actually, the function rotates the `target` object which is then followed
        by the `drone` (inside V-REP).

        `sensor_offset`: position of the sensor relative to the drone. Needed for a
        better azimuth value.
        """
        GOOD = azimuth = 5 # Degrees
        while abs(azimuth) >= GOOD:
            euler = self._target.get_orientation()
            __, azimuth, __ = goal.get_spherical(self._body, self.sensor_offset)
            correction_angle = self._rotation_pid.control(azimuth)
            if __debug__:
                print("Adjusting orientation...", correction_angle)
            euler[2] += radians(correction_angle) # euler[2] = Yaw
            self._target.set_orientation(euler)
            sleep(1)
        else:
            if __debug__:
                print("...Adjusted. Goal at {}°".format(azimuth))
            self._rotation_pid.reset()
            self.stabilize() # Wait for the drone to stabilize on the new angle

    def lock(self, goal: VRepObject):
        __, __, elevation = goal.get_spherical(self._body, self.sensor_offset)
        if abs(elevation) > self.MAX_ANGLE:
            self.altitude_adjust(goal)
        self.rotate_towards(goal)

    def stabilize(self):
        EPS = 0.001
        if __debug__:
            print("UAV stabilization in progress...")
        while True:
            lin_v, ang_v = self._body.get_velocity()
            if all(i < EPS for i in lin_v) and all(i < EPS for i in ang_v):
                if __debug__:
                    print("...done.")
                return
            else:
                sleep(0.05)

    def step_towards(self, goal: VRepObject):
        target_pos = self._target.get_position()
        correction = self._pid.control(-self._target.get_position(goal))
        self.total_distance += np.linalg.norm(correction)
        self._target.set_position(target_pos + correction)