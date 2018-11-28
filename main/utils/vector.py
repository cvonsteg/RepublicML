import math
import numpy as np

# Parking Vector class for now, as building all magic data model methods (__add__, __iter__, __sub__, etc.) is probably a project by itself.  

class Vector:
    """Model base class with fundamental statistical summarisation functionality"""
    def __init__(self, *args):
        if not args:
            self.vector = (0, 0)
        else:
            self.vector = args

    def magnitude(self):
        """Returns the scalar length of a vector"""
        return math.sqrt(sum(i**2 for i in self.vector))

    def mean(self):
        """Calculates the mean/average of a vector of numbers"""
        return np.nanmean(self.vector)

    def median(self):
        """Calculates the central value of a vector of numbers"""
        return np.nanmedian(self.vector)

    def variance(self):
        """Returns variance"""
        return np.nanvar(self.vector)

    def st_dev(self):
        """Returns standard deviation of vector"""
        return np.nanstd(self.vector)
