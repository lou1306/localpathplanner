# localpathplanner
Goal-oriented obstacle avoidance for multirotor UAVs (Master Thesis project)

This is a reactive, mapless obstacle avoidance algorithm for multirotor Unmanned Aerial Vehicles (UAVs).

The algorithm applies image processing techniques and qualitative evaluations
on a depth map acquired by the UAV in order to assess the reachability of the destination,
and establishes an alternative path whenever obstacles are detected.

The algorithm is not a local path planner in the strict sense of the word, since we do not prove
the optimality of the resulting path.

## Details on the implementation

The current implementation is a set of Python3 scripts which targets a 
[V-REP](http://www.coppeliarobotics.com/) simulation.
Example scenarios are stored in the `vrep_scenes` directory.

The scripts require opencv-python. On most \*NIX systems, a simple

    pip3 install opencv-python

should be enough. Please refer to the [opencv-python documentation](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html)
for detailed setup instructions for Lunux/MacOS and Windows.

After cloning the repository, the user should put the relevant V-REP API libraries in its main directory
(follow the instructions [here](http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm#python)).

V-REP must also be allowed to communicate on a TCP port (default 11111).

Then:

1. Start the simulation inside V-REP;
2. Start the `main.py` script with `python3 main.py`.

## Limitations

At the moment the implementation does not support the "wall-following" scenario, 
i.e. a situation where most of the field of view is obstructed by an obstacle.
This situation requires a different algorithm, such as the one proposed in [2].

## Further reading

* [My Master's Thesis (Italian only)](https://github.com/lou1306/localpathplanner/blob/master/docs/tesi.pdf)

* Slides from my Master's dissertation: [Italian](https://github.com/lou1306/localpathplanner/blob/master/docs/presentazione.pdf), [English](https://github.com/lou1306/localpathplanner/blob/master/docs/presentazione-EN.pdf)

* The [poster](https://github.com/lou1306/localpathplanner/blob/master/docs/poster.pdf) we presented at [COSIT2017](http://www.cosit2017.org/)

## Related work

[1] S. Hrabar, “Reactive obstacle avoidance for rotorcraft UAVs,” IEEE Int. Conf. Intell. Robot. Syst., no. August, pp. 4967–4974, 2011.

[2] T. Merz and F. Kendoul, “Beyond visual range obstacle avoidance and infrastructure inspection by an autonomous helicopter,” IEEE Int. Conf. Intell. Robot. Syst., no. August 2016, pp. 4953–4960, 2011.
