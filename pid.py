from __future__ import division
import numpy as np

class PID():
    """
    Simple PID controller.
    """
    def __init__(self, Kp, Ki, Kd=0, n=1, max_norm=0, max_int=np.inf):
        #type: (float, float, float, int, float, float)

        """
        Initializes the controller.

        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param n: Dimension of the state space
        :param max_norm: Maximum norm of the control vector. Disabled by default.
        :param max_int: Upper bound for each term of the integral accumulator. Disabled by default.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dim = n
        self.I_err = np.zeros(n)
        self.last_err = np.zeros(n)
        self.max_norm = max_norm
        self.MAX_INTEGRATOR = max_int * np.ones(n)

    def reset(self):
        """
        Resets the controller memory.
        """
        self.I_err = np.zeros(self.dim)
        self.last_err = np.zeros(self.dim)

    def control(self, err):
        """
        Returns a control input based on the error.
        :param err: The error vector, defined as (desired state - current state)
        :return: a correction vector that should be "added" to the current state.
        """
        if np.all(self.I_err < self.MAX_INTEGRATOR):
            self.I_err = np.add(self.I_err, err)
        diff = err - self.last_err
        self.last_err = err
        ctrl =  self.Kp * err + self.Ki * self.I_err + self.Kd * diff
        ctrl_norm = np.linalg.norm(ctrl)
        if ctrl_norm > self.max_norm > 0:
            ctrl *= self.max_norm/ctrl_norm
        return np.around(ctrl, decimals=3)

