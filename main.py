from _hashlib import new
from datetime import datetime
from time import sleep

import cv2
import numpy as np

from controllers import can_reach
from functions import pinhole_projection, inv_pinhole_projection, yaw_rotation, pitch_rotation, Direction
from pid import PID
from vrep_object import VRepClient
from drone import Drone

def radius(dist):
    return max(int(150 // dist), 1)


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

    sign = lambda x, y: 2 * int(x < y) - 1
    """sign(x,y) = -1 iff x<=y; 1 otherwise"""
    sign_x = sign(start[0], end[0])
    sign_y = sign(start[1], end[1])
    if start[0] == end[0]:
        return ((start[0], i) for i in range(start[1], end[1] + sign_y, sign_y))
    else:
        if start[1] == end[1]:
            return ((i, start[1]) for i in range(start[0], end[0] + sign_x, sign_x))
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


def depth_based_dilation(im):
    """Dilates a float image according to pixel depth.

    The input image is sliced by pixel intensity: (1, 0.9], (0.9, 0.8] etc.
    Each slice is dilated by a kernel which grows in size as the values get smaller.
    The slices are fused back together, lower slice overwrite higher ones.
    """
    acc = np.ones_like(im)

    for i in np.arange(1, 0.1, -0.1):
        im_slice = im.copy()
        im_slice[(im_slice <= i - 0.1) | (im_slice > i)] = 0

        ksize = 2 * radius((i - 0.1) * MAX_DEPTH)

        ker = np.ones((ksize, ksize), np.uint8)
        im_slice = cv2.dilate(im_slice, ker)

        # Replace "older" values
        acc = np.where(im_slice != 0, im_slice, acc)
    return acc


#############################################################

# V-REP Server address
HOST = "127.0.0.1"
PORT = 11111

# Maximum depth sensed by the camera
MAX_DEPTH = 10
EPS = 0.2
MAX_ANGLE = 45

RADIUS = 0.5

camera_settings = {
    "x_res": 256,
    "y_res": 256,
    "x_angle": 45,
    "y_angle": 45
}

pid = PID(4.5, 0.01, 0.1, 3, 0.15, max_int=10)
total_distance = 0

# Init connection and objects
client = VRepClient(HOST, PORT)

goal = client.get_object("Goal")
target = client.get_object("Quadricopter_target")
drone = client.get_object("Quadricopter_base")
sensor = client.get_depth_sensor("SR4000_sensor")

# Save real goal position for later
end_goal = goal.get_position()

# Main control loop

new_drone = Drone(client)
new_drone.lock(goal)

while True:
    # Get current data from server
    goal_pos = goal.get_position()
    target_pos = target.get_position()
    sensor_offset = - drone.get_position(sensor)
    delta = goal.get_position(drone)

    h_dist = np.linalg.norm(delta[0:2])
    v_dist = abs(delta[2])

    if h_dist < EPS:
        print("Goal reached! Total distance: {} m".format(total_distance))
        if not np.array_equal(goal_pos, end_goal):
            #client.create_dummy(goal.get_position(), 0.5)
            goal.set_position(end_goal)
            pid.reset()
            new_drone.lock(goal)
        else:
            break


    #reachable, d, min_depth, mask = can_reach(goal, target, drone, sensor_offset, sensor)
    reachable, d, min_depth, mask = new_drone.can_reach(goal)
    cv2.imshow('view', d)
    cv2.waitKey(1)

    if reachable:
        print("Reachable,", h_dist)
        # Goal is reachable!
        target_pos = target.get_position()
        correction = pid.control(-target.get_position(goal))
        total_distance += np.linalg.norm(correction)
        target.set_position(target_pos + correction)
    elif abs(h_dist - min_depth) < RADIUS:
        if np.array_equal(goal_pos, end_goal):
            print("Goal considered unsafe.")
            break
        else:
            print("Temporary goal considered unsafe. Re-evaluating...")
            goal.set_position(end_goal)
    else:
        new_drone.stabilize()
        # Mark current position and old waypoint
        # client.create_dummy(goal.get_position(), 0.5)
        # client.create_dummy(target.get_position(), 0.2)

        t = datetime.now()
        dist, azimuth, elevation = goal.get_spherical(drone, sensor_offset)
        X, Y = pinhole_projection(azimuth, elevation)

        light_zone = depth_based_dilation(d)
        avg_depth = min(0.999, ((min_depth + h_dist) / 2) / MAX_DEPTH)
        light_zone[light_zone <= avg_depth] = 0
        light_zone[light_zone > avg_depth] = 1

        distances = cv2.distanceTransform(light_zone.astype(np.uint8), cv2.DIST_L1, 3)
        candidates = np.column_stack(np.nonzero(distances == 1))
        if not candidates.size:
            print("No valid point for current view.")
            cv2.imshow('view', (depth_based_dilation(d) * 255).astype(np.uint8))
            print(min_depth, h_dist)
            print(avg_depth)
            # TODO wall-following algorithm
            goal.set_position(end_goal)
            break
        else:
            Y_p, X_p = min(candidates, key=lambda x: np.linalg.norm(np.array([Y, X]) - x) + 0.1 * abs(Y - x[0]))
            new_azimuth, new_elevation = inv_pinhole_projection(X_p, Y_p)

            # Invert the Y coordinates since images use a left-hand system:
            # (0,0) is top-left

            direction = Direction.get(np.array([X, -Y]), relative_to=np.array([X_p, -Y_p]))
            __, val = find_in_matrix(d, (Y_p, X_p), (Y, X), lambda x: x <= avg_depth)
            val = val or min_depth / MAX_DEPTH

            print("->", val)
            print("->", avg_depth)

            new_dist = min(val * MAX_DEPTH + RADIUS, h_dist)

            # TODO check original depth map for depth @ X_p, Y_p
            if d[Y_p, X_p] < 1 and d[Y_p, X_p] * MAX_DEPTH - new_dist < RADIUS:
                new_dist = d[Y_p, X_p] - RADIUS

            # Apply two 3d rotations to the unit vector so it points to the new goal
            unit_vec = np.array([1, 0, 0], np.float32)
            unit_vec = yaw_rotation(unit_vec, new_azimuth)
            unit_vec = pitch_rotation(unit_vec, new_elevation)

            new_delta = unit_vec * new_dist

            ### DEBUG LOGS, remove in production ####################
            print("REPLANNING TIME:", datetime.now() - t)
            d1 = np.array(d)
            for y, x in candidates:
                d1 = cv2.circle(d1, (x, y), 1, 0, 2)
            d1 = cv2.circle(d1, (X_p, Y_p), 7, 1, -1)

            print("OLD ", dist, azimuth, elevation)
            print("NEW ", new_dist, new_azimuth, new_elevation)
            print("min_depth:", min_depth)
            print("direction:", direction)

            #cv2.imshow('view', d1 + mask)
            #cv2.waitKey()

            ### END DEBUG LOGS ######################################
            goal.set_position(new_delta, drone)
            #new_drone.lock(goal)
            pid.reset()  # Because the goal has changed

cv2.destroyAllWindows()

# Close the connection
client.close_connection()
