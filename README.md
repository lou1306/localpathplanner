# localpathplanner
Local Path Planner for multirotor UAVs (Master Thesis project)

This is a reactive, mapless local path planner for multirotor UAVs.

The algorithm applies image processing techniques and qualitative evaluations
on a depth map acquired by the UAV in order to assess the reachability of the destination,
and establishes an alternative path when unforeseen obstacles are detected.

The current implementation targets a [V-REP](http://www.coppeliarobotics.com/) simulation.
