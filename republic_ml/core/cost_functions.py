import numpy as np

from .tensor import Tensor


class Cost:
    def cost(self, predicted: Tensor, actual: Tensor) -> float:
        """ Calculate loss """
        raise NotImplementedError


class MSE(Cost):
    """ Implement MeanSquaredError (MSE) """

    def cost(sef, predicted: Tensor, actual: Tensor) -> float:
        m = len(predicted)
        return (1 / (2 * m)) * sum(np.square(predicted - actual))


class MSEMixin:
    def cost(self) -> float:
        return (1 / (2 * self.m)) * np.sum(np.square(self.hypothesis - self.y))

    def d_theta0(self):
        return (1 / self.m) * np.sum(self.hypothesis - self.y)

    def d_theta1(self):
        return (1 / self.m) * np.sum((self.hypothesis - self.y) * self.x)
