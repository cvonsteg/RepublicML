import numpy as np
import matplotlib.pyplot as plt
from republic_ml.supervised_models.supervised_model import SupervisedModel


class LinearRegression(SupervisedModel):
    def __init__(self, x, y, alpha):
        super().__init__(x, y, alpha)

    def predict(self, new_x):
        """Predict value of new_x"""
        new_x = np.matrix([np.ones(len(new_x)), new_x]).T
        return self.hypothesis(new_x)

    def fit(self, method='gradient descent', n_iter=100, plot=False):
        """
        Conducts n_iter iterations of gradient descent steps to estimate theta.
        Initializes a dict object with keys representing calculated cost,
        and values representing corresponding theta coefficients.
        """
        if method == 'gradient descent':
            self.cost_dict = {}
            for i in np.arange(n_iter):
                cost = self.cost()
                self.cost_dict[cost] = self.theta
                self._gradient_descent_step()
                print(f"Theta: {self.theta}, Cost: {cost}")

            self.theta = self.cost_dict[min(self.cost_dict)]

        else:
            self.theta = self._normal_equation()

        if plot:
            plt.plot(
                [i for i in np.arange(n_iter)],
                [v for v in self.cost_dict.keys()]
            )
            plt.title("Cost vs gradient descent iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.show()

    def hypothesis(self, x):
        """
        Returns hypothesis, also known as a prediction for a given
        input value of x.
        """
        return np.dot(x, self.theta.T)

    def cost(self):
        """
        Represents how large the error is between the hypothesis and labelled
        examples.
        Cost function (J(theta)) to be minimized during regression:
            J(theta) = 1 / 2m * ((h_theta(x) - y) ** 2)
        """
        const = (1 / (2 * self.m))
        square_err = np.square((self.hypothesis(self.x) - self.y))
        cost = const * ((square_err).sum())
        return cost

    def _gradient_descent_step(self):
        """
        Batch Gradient Descent equation to calculate theta:
            theta_i := theta_i - (alpha / m) * ((h_theta(x) - y) * x)

        """
        err = self.hypothesis(self.x) - self.y
        del_j = np.dot(err.T, self.x)
        self.theta = self.theta - ((self.alpha / self.m) * del_j)

    def _normal_equation(self):
        """
        Attempts to solve for Theta coefficients using the normal
        equation:
            Theta = (X.T * X)^-1 * X.T * y
        """
        X_T = self.x.T
        X_inv = np.invert(np.dot(X_T, self.x).astype(int))
        return X_inv * X_T * self.y
