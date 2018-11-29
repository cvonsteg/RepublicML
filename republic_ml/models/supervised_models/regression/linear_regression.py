import math
import numpy as np 


class LinRegSimple(object):
    """Simple Linear Regression class"""
    def __init__(self, x: np.array, y: np.array) -> None:
        self.x = x 
        self.y = y 

        if len(self.x) != len(self.y):
            self.x, self.y = self._truncate_vector_lenghts()

    def sum_of_squares(self):
        """Calculates sum of squares for both parameters"""
        sum_of_x = math.sqrt(self.x.sum())
        sum_of_y = math.sqrt(self.y.sum())
        return (sum_of_x, sum_of_y)

    def _truncate_vector_lenghts(self):
        """Finds the minimum length of the two vectors and truncates both to this length"""
        min_len = min(len(self.x), len(self.y))
        return self.x[:min_len], self.y[:min_len]
    
