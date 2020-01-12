from typing import Optional

import numpy as np
from core.cost_functions import MSE
from core.tensor import Tensor


class LinearRegression(MSEMixin):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        """All values initialised into long arrays/matrices"""
        # Before formatting, check input vectors
        self._check_input_vectors(x, y)
        # To allow matrix operations, constant value of 1
        # is being added to every row of input x vector
        self.x = self._format_x_matrix(x)
        self.y = np.matrix(y).T
        # Array of thetas initialized to 0
        self.theta = np.zeros(self.x.shape[1])
        # constant m represents number of input values
        self.m = self.x.shape[0]

    def predict(self, x: Tensor) -> Tensor:
        x = self._format_x_matrix(x)
        return self.hypothesis(x=x)

    def hypothesis(self, x: Optional[Tensor] = None) -> Tensor:
        """
        Hypothesis is equivalent to the sum of x * theta,
        in matrix terms, this equates to the dot product
        """
        if x is None:
            x = self.x
        return np.dot(x, self.theta.T)

    def fit(self, method="gradient_descent", n_iter=100) -> None:
        """
        Fit theta to input values of x using cost function defined in method
        """
        pass

    def cost(self) -> float:
        """
        Calculate loss/cost of current hypothesis.
        For purposes of linear regression, this will be done using MSE
        """
        return MSE().cost(predicted=self.hypothesis(), actual=self.y)

    @staticmethod
    def _format_x_matrix(x) -> Tensor:
        """ Formats a array of x values into a formatted matrix """
        return np.matrix([np.ones(len(x)), x]).T

    @staticmethod
    def _check_input_vectors(x: Tensor, y: Tensor):
        """ Ensures x and y vectors are of same length """
        len_x, len_y = len(x), len(y)
        assert (
            len_x == len_y
        ), f"Input vectors must be of same length.  Provided vectors have lengths {len_x}, {len_y} respectively"
