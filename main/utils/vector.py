import numpy as np  
import math

class Vector:
    """Model base class with fundamental statistical summarisation functionality"""
    def __init__(self, *args):
        if len(args) == 0:
            self.vector = (0,0)
        else:
            self.vector = args

        self.mean = self._mean()
        self.median = self._median()
        self.variance = self._variance()
        self.stdev = self._st_dev()
    
    def __len__(self):
        """Returns the scalar length of a vector"""
        return math.sqrt(sum(i**2 for i in self.vector))

    def _mean(self):
        """Calculates the mean/average of a vector of numbers"""
        return np.nanmean(self.vector)

    def _median(self):
        """Calculates the central value of a vector of numbers"""
        return np.nanmedian(self.vector)

    def _variance(self):
        """Returns variance"""
        return np.nanvar(self.vector)

    def _st_dev(self):
        """Returns standard deviation of vector"""
        return np.nanstd(self.vector)
