import math
import numpy as np


class Regression(object):
    """Simple Linear Regression class"""
    def __init__(self, x: np.array, y: np.array) -> None:
        self.x = x
        self.y = y
        self.slope = None
        self.y_int = None
        self.y_hat = None

        if len(self.x) != len(self.y):
            self.x, self.y = self._truncate_vector_lenghts()

    def predict(self, x_hat):
        """Calculates predicted y-value given regression coeffs"""
        return (self.slope * x_hat) + self.y_int

    def fit(self):
        """Fits the model to the data, calculating slope and intercept coefficients"""
        self.slope = self._slope()
        self.y_int = self._y_intercept()
        print('Slope and Intercept calculated - model ready to use')

    def coefficients(self):
        """Returns regression coefficients"""
        print(" --- Regression Coefficients --- ")
        print(" --- Slope: {}".format(self._slope()))
        print(" --- Intercept: {}".format(self._y_intercept()))

    def sum_of_squares(self):
        """Calculates sum of squares for both parameters"""
        sum_of_x = math.sqrt(sum(self.x ** 2))
        sum_of_y = math.sqrt(sum(self.y ** 2))
        return sum_of_x, sum_of_y

    def rmse(self):
        """Calculates root mean squared error of model"""
        self.y_hat = self.predict(self.x)
        return (sum((self.y - self.y_hat) ** 2) / len(self.x)) ** 0.5

    def _slope(self):
        """Calculate slope of line"""
        n = len(self.x)
        sigma_y = self.y.sum()
        sigma_x = self.x.sum()
        sigma_xy = sum(self.x * self.y)
        sigma_x_sq = sum(self.x ** 2)
        slope = ((n * sigma_xy) - (sigma_x * sigma_y)) / ((n * sigma_x_sq) - (sigma_x ** 2))
        return slope

    def _y_intercept(self):
        """Calculate y-intercept of line"""
        n = len(self.x)
        sigma_y = self.y.sum()
        sigma_x = self.x.sum()
        sigma_xy = sum(self.x * self.y)
        sigma_x_sq = sum(self.x ** 2)
        intercept = ((sigma_y * sigma_x_sq) - (sigma_x * sigma_xy)) / ((n * sigma_x) - (sigma_x ** 2))
        return intercept

    def _truncate_vector_lenghts(self):
        """Finds the minimum length of the two vectors and truncates both to this length"""
        min_len = min(len(self.x), len(self.y))
        return self.x[:min_len], self.y[:min_len]
