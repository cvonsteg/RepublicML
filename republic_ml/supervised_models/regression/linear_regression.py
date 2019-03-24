import numpy as np
from republic_ml.supervised_models.supervised_model import SupervisedModel


class LinearRegression(SupervisedModel):
    def __init__(self, x, y, alpha):
        super().__init__(x, y, alpha)

    def predict(self, new_x):
        """Predict value of new_x"""
        new_x = np.matrix([np.ones(len(new_x)), new_x]).T
        return self.hypothesis(new_x)

    def fit(self, n_iter):
        """Execute gradient descent algorithm to estimate theta coefficients"""
        # Alther values of theta
        # call cost_function() to caluclate cost
        # TO MAKE MORE EFFICIENT: could make generator which only store minimum
        # cost locally. but for plotting purposes this works well

        # # TODO: WHILE COST > 0
        self.cost_dict = {}
        for i in np.arange(n_iter):
            cost = self.cost()
            self.cost_dict[cost] = self.theta
            self._gradient_descent_step()
            print(f"Theta: {self.theta}, Cost: {cost}")

        self.theta = self.cost_dict[min(self.cost_dict)]

    def hypothesis(self, x):
        return np.dot(x, self.theta.T)

    def cost(self):
        """
        Function to be minimized during regression:
            J(theta) = 1 / 2m * SUM((h_theta(x) - y)^2)
        """
        cost = (1 / 2 * self.m) * np.square(self.hypothesis(self.x) - self.y).sum()
        return cost

    def _gradient_descent_step(self):
        """Batch Gradient Descent equation to calculate theta"""
        # h_theta = np.dot(self.theta.T, self.x)
        # err = h_theta - self.y
        # del_j = np.dot(err, self.x)
        # theta_new = theta - ((alpha / m) * (del_j))
        err = self.hypothesis(self.x) - self.y
        del_j = np.dot(err.T, self.x)
        self.theta = self.theta - ((self.alpha / self.m) * del_j)
