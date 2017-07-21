from datetime import datetime
from vrep_object import VRepClient
from drone import Drone

import cv2
import numpy as np

from functions import (pinhole_projection, inv_pinhole_projection,
                       apinhole_projection, ainv_pinhole_projection,
                       yaw_rotation, pitch_rotation,
                       find_in_matrix)


def radius(distance):
    return max(int(120 // distance), 1)


def depth_based_dilation(im):
    """Dilates a float image according to pixel depth.

    The input image is sliced by pixel intensity: (1, 0.9], (0.9, 0.8] etc.
    Each slice is dilated by a kernel which grows in size as the values
    get smaller.
    The slices are fused back together, lower slice overwrite higher ones.
    """
    acc = np.ones_like(im)

    for i in np.arange(1, 0.1, -0.1):
        im_slice = im.copy()
        im_slice[(im_slice <= i - 0.1) | (im_slice > i)] = 0

        ker_size = 2 * radius((i - 0.1) * MAX_DEPTH)

        ker = np.ones((ker_size, ker_size), np.uint8)
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

tries = 0

# Init connection and objects
client = VRepClient(HOST, PORT)

# Save target position for later
goal = client.get_object("Goal")
end_goal = goal.get_position()

print("Local Path Planner is ready.")

drone = Drone(client)
drone.lock(goal)

start_time = datetime.now()

# Main control loop
while True:
    # Get current data from server
    delta = goal.get_position(drone)
    h_dist = np.linalg.norm(delta[0:2])
    v_dist = abs(delta[2])

    if h_dist < EPS:
        goal_pos = goal.get_position()
        if not np.array_equal(goal_pos, end_goal):
            if __debug__:
                print("Goal reached! Total distance: {} m".format(drone.total_distance))
            goal.set_position(end_goal)
            drone.reset_controllers()
            drone.lock(goal)
            continue
        else:
            print("Goal reached! Total distance: {} m".format(drone.total_distance))
            print("Total time:", datetime.now() - start_time)
            break

    reachable, d, min_depth, mask = drone.can_reach(goal)

    if reachable:
        if __debug__:
            print("Reachable,", h_dist)
        drone.step_towards(goal)
    elif abs(h_dist - min_depth) < RADIUS:
        goal_pos = goal.get_position()
        if np.array_equal(goal_pos, end_goal):
            print("Goal considered unsafe. I'm afraid I can't do that.")
            break
        else:
            print("Temporary goal considered unsafe. Re-evaluating...")
            goal.set_position(end_goal)
    else:
        drone.stabilize()
        # Mark current position and old waypoint
        # client.create_dummy(goal.get_position(), 0.5)
        # client.create_dummy(target.get_position(), 0.2)

        t = datetime.now()
        dist, azimuth, elevation = goal.get_spherical(drone,
                                                      drone.sensor_offset)
        X, Y = pinhole_projection(azimuth, elevation)
        X1, Y1 = apinhole_projection(azimuth, elevation)

        print(X, Y, "\n", X1, Y1, drone.sensor_offset)

        # Dilation + Threshold
        light_zone = depth_based_dilation(d)
        avg_depth = min(0.999, ((min_depth + h_dist) / 2) / MAX_DEPTH)
        light_zone[light_zone <= avg_depth] = 0
        light_zone[light_zone > avg_depth] = 1

        # Distance Transform + Search for optimal pixel
        distances = cv2.distanceTransform(light_zone.astype(np.uint8), cv2.DIST_L1, 3)
        candidates = np.column_stack(np.nonzero(distances == 1))
        if not len(candidates):
            tries += 1
            if tries > 2:
                print("No valid point for current view.")
                # TODO wall-following algorithm
                goal.set_position(end_goal)
                drone.escape(goal)
                break
            else:
                continue
        else:
            tries = 0
            X_p, Y_p = min(candidates,
                           key=lambda c:
                           np.linalg.norm(np.array([X, Y]) - c) +
                           0.1 * abs(X - c[0]))
            new_azimuth, new_elevation = inv_pinhole_projection(X_p, Y_p)
            az1, ev1 = ainv_pinhole_projection(X_p, Y_p)

            print(new_azimuth, new_elevation, "\n", az1, ev1)

            # Invert the Y coordinates since images use a left-hand system:
            # (0,0) is top-left

            __, val = find_in_matrix(d, (X_p, Y_p), (X, Y), lambda depth: depth <= avg_depth)
            val = val or min_depth / MAX_DEPTH

            new_dist = min(val * MAX_DEPTH + RADIUS, h_dist)

            # check original depth map for depth @ X_p, Y_p
            if d[X_p, Y_p] < 1 and d[X_p, Y_p] * MAX_DEPTH - new_dist < RADIUS:
                new_dist = d[X_p, Y_p] - RADIUS

            # Apply two 3d rotations to the unit vector so it points to the new goal
            unit_vec = np.array([1, 0, 0], np.float32)
            unit_vec = yaw_rotation(unit_vec, new_azimuth)
            unit_vec = pitch_rotation(unit_vec, new_elevation)

            new_delta = unit_vec * new_dist
            goal.set_position(new_delta, drone)
            goal.duplicate()
            drone.reset_controllers()

            if __debug__:
                print("REPLANNING TIME:", datetime.now() - t)
                d1 = np.array(d)
                for y, x in candidates:
                    d1 = cv2.circle(d1, (x, y), 1, 0, 2)
                d1 = cv2.circle(d1, (X_p, Y_p), 7, 1, -1)

                print("OLD ", dist, azimuth, elevation)
                print("NEW ", new_dist, new_azimuth, new_elevation)
                print("min_depth:", min_depth)

cv2.destroyAllWindows()

# Close the connection
client.close_connection()
