from datetime import datetime
from vrep_object import VRepClient
from drone import Drone

import cv2
import numpy as np

from functions import (pinhole_projection, inv_pinhole_projection,
                       yaw_rotation, pitch_rotation,
                       find_in_matrix)

#############################################################

# V-REP Server address
HOST = "127.0.0.1"
PORT = 11111

EPS = 0.2 # meters
UNIT_VEC = np.array([1, 0, 0], np.float32)

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
                print("Goal reached! Total distance: {} m".format(
                    drone.total_distance))
            goal.set_position(end_goal)
            drone.reset_controllers()
            drone.lock(goal)
            continue
        else:
            print("Goal reached! Total distance: {} m".format(
                drone.total_distance))
            print("Total time:", datetime.now() - start_time)
            break

    reachable, d, min_depth, mask = drone.can_reach(goal)

    if reachable:
        if __debug__:
            print("Reachable,", h_dist)

        drone.step_towards(goal)
    elif abs(h_dist - min_depth) < drone.radius:
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

        # Dilation + Threshold
        light_zone = drone.sensor.get_dilated_depth_buffer(drone.radius_to_pixels)
        avg_depth = min(0.999, ((min_depth + h_dist) / 2) / drone.sensor.max_depth)
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
            Y_p, X_p = min(candidates,
                           key=lambda c:
                           np.linalg.norm(np.array([Y, X]) - c) +
                           0.1 * abs(Y - c[0]))
            new_azimuth, new_elevation = inv_pinhole_projection(X_p, Y_p)

            __, val = find_in_matrix(d, (X_p, Y_p), (X, Y), lambda depth: depth <= avg_depth)
            val = val or min_depth / drone.sensor.max_depth

            new_dist = min(val * drone.sensor.max_depth + drone.radius, h_dist)

            # check original depth map for depth @ X_p, Y_p
            if d[X_p, Y_p] < 1 and d[X_p, Y_p] * drone.sensor.max_depth - new_dist < drone.radius:
                new_dist = d[X_p, Y_p] * drone.sensor.max_depth - drone.radius

            # Apply two 3d rotations to the unit vector so it points to the new goal
            unit_vec = yaw_rotation(UNIT_VEC, new_azimuth)
            unit_vec = pitch_rotation(unit_vec, new_elevation)

            new_delta = unit_vec * new_dist
            goal.set_position(new_delta, drone)
            goal.duplicate()
            drone.reset_controllers()

            if __debug__:
                print("REPLANNING TIME:", datetime.now() - t)
                # # Uncomment to write the current depth map to disk
                # d1 = np.array(d)
                # for y, x in candidates:
                #     d1 = cv2.circle(d1, (x, y), 1, 0, 2)
                # d1 = cv2.circle(d1, (X_p, Y_p), 7, 128, -1)
                # cv2.imwrite("{}.png".format(datetime.now()), d1 * 256)

                print("OLD ", dist, azimuth, elevation)
                print("NEW ", new_dist, new_azimuth, new_elevation)
                print("min_depth:", min_depth)

# Close the connection
client.close_connection()
