import numpy as np


class Hypothesis:
    def __init__(self, x):
        """
        Initializes H_theta(x) object.  To enable matrix/vector operations, x 
        array is initialized with an 
        """
        ones = np.ones(len(x))
        self.x = np.matrix([ones, x])
        self.theta = np.zeros(len(x))

