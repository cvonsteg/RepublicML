import numpy as np


class SupervisedModel:
    def __init__(self, x, y, alpha) -> None:
        self.y = np.matrix(y).T
        self.x = np.matrix([np.ones(len(x)), x]).T
        self.m = self.x.shape[0]
        self.theta = np.matrix(np.zeros(self.x.shape[1]))
        self.alpha = alpha

    def predict(self, unknown_var):
        pass

    def fit(self):
        pass

    def hypothesis(self, x):
        """
        Returns hypothesis, also known as a prediction for a given
        input value of x.
        """
        pass

    def cost(self):
        """
        Represents how large the error is between the hypothesis and labelled
        examples.
        Cost function (J(theta)) to be minimized during regression:
            J(theta) = 1 / 2m * ((h_theta(x) - y) ** 2)
        """
        pass

    def _gradient_descent_step(self):
        """
        Batch Gradient Descent equation to calculate theta:
            theta_i := theta_i - (alpha / m) * ((h_theta(x) - y) * x)

        """
        pass
