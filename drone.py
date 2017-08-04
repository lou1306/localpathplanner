from enum import Enum
from math import radians, sin, cos
from time import sleep

import cv2
import numpy as np

from functions import pinhole_projection
from pid import PID
from vrep_object import VRepClient, VRepObject


class Visibility(Enum):
    VISIBLE = 1
    NOT_VISIBLE = 2
    UNREACHABLE = 3


class Drone(VRepObject):
    SAFETY_RADIUS = 0.3  # meters. Value to add to the real radius of the UAV

    def __init__(self, client: VRepClient):
        self._body = client.get_object("Quadricopter_base")
        self._model = client.get_object("Quadricopter")
        super().__init__(client.id, self._body.handle, "")
        self._target = client.get_object("Quadricopter_target")
        self.sensor = client.get_object("fast3DLaserScanner_sensor")

        self._rotation_pid = PID(0.2, 0.05, 0.2, 1, max_int=3)
        self._altitude_pid = PID(0.2, 0.02, 0.2, 1)
        self._pid = PID(4.5, 0.01, 0.1, 3, 0.15, max_int=10)

        self.total_distance = 0
        self.sensor_offset = self._body.get_position(self.sensor)
        self.radius = self._get_radius() + self.SAFETY_RADIUS

        # Base and height of the visibility cone (actually a pyramid)
        B = 2 * self.sensor.max_depth * sin(radians(self.sensor.angle))
        H = 2 * self.sensor.max_depth * cos(radians(self.sensor.angle))
        # Constant for pixel-to-meters conversion
        self.K = self.sensor.res
        if abs(B - H) > 1e-3:
            self.K *= H / B


    def _get_radius(self) -> float:
        """Return the effective radius of the drone

        The radius is half of the distance between the extrema of the model's
        bounding box.
        """
        bbox = self._model.get_bbox()
        bbox_span = np.linalg.norm(bbox[1]-bbox[0])
        return bbox_span / 2

    def altitude_adjust(self, goal: VRepObject) -> None:
        good = err = 0.5  # meters
        while abs(err) >= good:
            goal_pos = goal.get_position(self._body)
            err = goal_pos[2]  # z-coordinate
            correction = self._altitude_pid.control(err)
            if __debug__:
                print("Adjusting altitude...", correction)
            self._target.set_position(self._target.get_position() +
                                      np.array([0, 0, correction]))
            self.total_distance += np.linalg.norm(correction)
            sleep(1)
        else:
            if __debug__:
                print("...Adjusted. Goal at {} m".format(err))
            self._altitude_pid.reset()
            sleep(2)  # Wait for the drone to stabilize

    def can_reach(self, goal: VRepObject):
        dist, azimuth, elevation = goal.get_spherical(self._body,
                                                      self.sensor_offset)
        delta = goal.get_position(self._target)
        h_dist = np.linalg.norm(delta[0:2])

        res, d = self.sensor.get_depth_buffer()

        X, Y = pinhole_projection(azimuth, elevation)
        ball_r = self.radius_to_pixels(dist)
        mask = cv2.circle(np.zeros_like(d), (X, Y), ball_r, 1, -1)
        try:
            min_depth = np.min(d[mask == 1]) * self.sensor.max_depth
        except ValueError:
            # Mask has no white pixel. Center view on goal and retry
            self.lock(goal)
            return self.can_reach(goal)
        reachable = h_dist < 1 or dist - min_depth < -self.radius or \
            min_depth == self.sensor.max_depth
        return reachable, d, min_depth, mask

    def radius_to_pixels(self, dist: float) -> int:
        """Converts a drone radius in pixels, at the given distance.

        This function returns the size in pixels of a segment of length RADIUS,
        placed at distance `dist` and orthogonal to the principal axis of the
        camera.
        """
        return max(int(self.K * self.radius / dist), 1)

    def reset_controllers(self):
        self._pid.reset()
        self._altitude_pid.reset()
        self._rotation_pid.reset()

    def rotate_towards(self, goal: VRepObject):
        """Rotate the drone until it points towards the goal.

        Actually, the function rotates the `target` object which is then
        followed by the `drone` (inside V-REP).
        """
        good = azimuth = 2  # Degrees
        while abs(azimuth) >= good:
            euler = self._target.get_orientation()
            __, azimuth, __ = goal.get_spherical(self._body,
                                                 self.sensor_offset)
            correction_angle = self._rotation_pid.control(azimuth)
            if __debug__:
                print("Adjusting orientation...", correction_angle)
            euler[2] += radians(correction_angle)  # euler[2] = Yaw
            self._target.set_orientation(euler)
            sleep(1)
        else:
            if __debug__:
                print("...Adjusted. Goal at {}°".format(azimuth))
            self._rotation_pid.reset()
            self.stabilize()  # Wait for the drone to stabilize on new angle

    def lock(self, goal: VRepObject):
        __, azimuth, elevation = goal.get_spherical(self._body,
                                                    self.sensor_offset)
        X, Y = pinhole_projection(azimuth, elevation)
        if abs(elevation) > self.sensor.angle or not 0 <= Y < 256:
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

    def step_towards(self, goal: VRepObject):
        """Move the drone towards the goal.
        """
        target_pos = self._target.get_position()
        correction = self._pid.control(-self._target.get_position(goal))
        self.total_distance += np.linalg.norm(correction)
        self._target.set_position(target_pos + correction)

    def escape(self, goal):
        # TODO implement wall-following algorithm
        self.rotate(60)
        __, d = self.sensor.get_depth_buffer()
        left_space = len(d[d == 1])
        self.rotate(-120)
        __, d = self.sensor.get_depth_buffer()
        right_space = len(d[d == 1])
        go_left = left_space >= right_space

    def rotate(self, angle: float):
        """Perform an arbitrary yaw rotation.
        
        Args:
            angle (float): Yaw angle, in degrees. Positive = rotates left
        """
        self._rotation_pid.reset()
        while abs(angle) > 2:
            euler = self._target.get_orientation()
            correction = self._rotation_pid.control(angle)
            angle -= correction
            euler[2] += radians(correction)  # euler[2] = Yaw
            self._target.set_orientation(euler)
            sleep(1)
        self.stabilize()
