import numpy as np  

class BaseModel:
    """Model base class with basic functionality"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def mean(vector):
        """Calculates the mean/average of a vector of numbers"""
        return np.nanmean(vector)

    @staticmethod
    def median(vector):
        """Calculates the central value of a vector of numbers"""
        return np.nanmedian(vector)

    @staticmethod
    def variance(vector):
        """Returns variance"""
        return np.nanvar(vector)

    @staticmethod
    def st_dev(vector):
        """Returns standard deviation of vector"""
        return np.nanstd(vector)
