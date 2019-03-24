import numpy as np


class Regression:
    def __init__(self, x, y, alpha, n_iter):
        self.x = np.matrix([np.ones(len(x)), x])
        self.y = y
        self.m = self.x.shape[1]
        self.theta = np.zeros(self.m)
        self.alpha = alpha

    def predict(self, new_x):
        """Predict value of new_x"""
        pass

    def fit(self):
        """Execute gradient descent algorithm to estimate theta coefficients"""
        # Alther values of theta
        # call cost_function() to caluclate cost
        # TO MAKE MORE EFFICIENT: could make generator which only store minimum
        # cost locally. but for plotting purposes this works well
        costs = {}
        for i in np.arange(n_iter):
            new_theta = self._gradient_descent()
            t, c = self._calculate_cost(new_theta)
            costs[t] = c

    def _clalculate_cost(self, theta):
        """
        Function to be minimized during regression:
            J(theta) = 1 / 2m * SUM((h_theta(x) - y)^2)
        """
        cost = (1 / 2 * self.m) * np.square(np.dot(theta, self.x) - self.y).sum()
        return theta, cost

    def _gradient_descent(self):
        """Batch Gradient Descent equation to calculate theta"""
        # h_theta = np.dot(self.theta.T, self.x)
        # err = h_theta - self.y
        # del_j = np.dot(err, self.x)
        # theta_new = theta - ((alpha / m) * (del_j))
