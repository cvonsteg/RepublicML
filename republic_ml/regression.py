import numpy as np

from republic_ml.core.cost_functions import MSEMixin
from republic_ml.core.tensor import Tensor


class LinearRegression(MSEMixin):
    """
    Class for training and predicting values via simple linear regression

    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        """
        Parameters
        ----------
        x : Tensor
            Tensor of input features
        y : Tensor
            Tensor of expected outputs for given input values
        """
        # Before formatting, check input vectors
        self._check_input_vectors(x, y)
        # To allow matrix operations, constant value of 1
        # is being added to every row of input x vector
        self.x = self._format_x_matrix(x)
        self.y = y
        # Array of thetas initialized to 0
        self.theta = np.zeros(self.x.shape[1])
        # constant m represents number of input values
        self.m = self.x.shape[0]

    def predict(self, x: Tensor) -> Tensor:
        """
        Predict values for unseen inputs

        Parameters
        ----------
        x : Tensor
            Tensor array of new values to be predicted

        Returns
        -------
        Tensor
            Tensor arrray of new predicted values
        """
        x = self._format_x_matrix(x)
        return self._dot_product(x, self.theta.T)

    @property
    def hypothesis(self) -> Tensor:
        """
        Hypothesis is equivalent to the sum of x * theta,
        in matrix terms, this equates to the dot product
        """
        return self._dot_product(self.x, self.theta.T)

    def fit_gd(self, lr: float = 0.001, epoch=100) -> None:
        """
        Fit theta to input values of x using gradient descent

        Parameters
        ----------
        lr : float
            Learning Rate, which determines how large of a step the GD algorithm takes

        epoch : int
            Number of epoch iterations to use for coefficient fitting
        """
        print(f"Starting coefficients: {self.theta}")
        for epoch in range(epoch):
            print(f"Epoch: {epoch}")
            print(f"Cost: {self.cost()}")
            theta_0_new = self.theta[0] - (lr * self.d_theta0())
            theta_1_new = self.theta[1] - (lr * self.d_theta1())
            self.theta[0], self.theta[1] = theta_0_new, theta_1_new
            print(f"Coefficients: {self.theta}")

    @staticmethod
    def _dot_product(x: Tensor, y: Tensor) -> Tensor:
        """
        Return the dot product of 2 tensors
        """
        return np.dot(x, y)

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
