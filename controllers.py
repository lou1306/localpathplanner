from math import radians
from time import sleep

import numpy as np
import cv2

from functions import sph2cart, pinhole_projection, inv_pinhole_projection, yaw_rotation, pitch_rotation, Direction
from pid import PID

def radius(dist):
    return max(int(120 // dist), 1)

def rotate_towards(goal, target, drone, offset):
    """Rotates the drone until it points towards the goal.

    Actually the function rotates the `target` object which is then followed
    by the `drone` (inside V-REP).

    `sensor_offset`: position of the sensor relative to the drone. Needed for a
    better azimuth value.
    """
    GOOD = 5 # Degrees
    azimuth = 2*GOOD
    rotation_pid = PID(0.3, 0.05, 0.5, 1, max_int=3)

    while abs(azimuth) > GOOD:
        try:
            euler = target.get_orientation()
            __, azimuth, __ = goal.get_spherical(drone.handle, offset)
        except ConnectionError:
            continue
        correction_angle = rotation_pid.control(azimuth)
        print("Adjusting orientation...", correction_angle)
        euler[2] += radians(correction_angle) # euler[2] = Yaw 
        try:
            target.set_orientation(euler)
            sleep(1)
        except ConnectionError:
            continue
    else:
        print("...Adjusted. Goal at {}°".format(azimuth))
        rotation_pid.reset()
        sleep(10) # Wait for the drone to stabilize on the new angle

def altitude_adjust(goal, target, drone):
    altitude_pid = PID(0.2, 0.02, 0.2, 1)
    GOOD = 0.5 # meters
    err = 2*GOOD
    while abs(err) > GOOD:
        try:
            goal_pos = goal.get_position(target.handle)
            err = goal_pos[2] # z-coordinate
            correction = altitude_pid.control(err)
            print("Adjusting altitude...", correction)
            target.set_position(target.get_position() +
                np.array([0,0, correction]))
            sleep(1)
        except ConnectionError:
            continue
    else:
        print("...Adjusted. Goal at {} m".format(err))
        altitude_pid.reset()
        sleep(2) # Wait for the drone to stabilize



camera_settings = {
    "x_res": 256,
    "y_res": 256,
    "x_angle": 45,
    "y_angle": 45
}
MAX_DEPTH = 10

def can_reach(goal, target, drone, sensor_offset, sensor):
    # Rotate the drone if target out of FOV
    MAX_ANGLE = 45
    while True:
        try:
            dist, azimuth, elevation = goal.get_spherical(drone.handle, sensor_offset)
            delta = goal.get_position(target.handle)
            break
        except ConnectionError:
            continue
    X, Y = pinhole_projection(azimuth, elevation, camera_settings)
    h_dist = np.linalg.norm(delta[0:2])
    v_dist = delta[2]

    if abs(elevation) > MAX_ANGLE:
        altitude_adjust(goal, target, drone)

    if  dist > 1 and (
        abs(azimuth) > MAX_ANGLE 
        or not (0<= X <= camera_settings["x_res"])
        ):
        rotate_towards(goal, target, drone, sensor_offset)
    while True:
        try:
            res, d = sensor.get_depth_buffer()
            d = np.array(d, np.float32).reshape((res[1], res[0]))
            d = np.flipud(d) # the depth buffer is upside-down
            dist, azimuth, elevation = goal.get_spherical(drone.handle, sensor_offset)
            break
        except ConnectionError as exc:
            continue
    X, Y = pinhole_projection(azimuth, elevation, camera_settings)
    ball_r = radius(dist)
    mask = cv2.circle(np.zeros_like(d), (X,Y), ball_r, 1, -1)

    try:
        min_depth = np.min(d[mask==1]) * MAX_DEPTH
    except ValueError:
        # Mask has no white pixel.
        min_depth = d[Y,X] * MAX_DEPTH
    return h_dist < 1 or dist - min_depth < -0.5 or min_depth == MAX_DEPTH, d, min_depth, mask