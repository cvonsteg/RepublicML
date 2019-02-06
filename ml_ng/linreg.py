import numpy as np


class Regression:
    def __init__(self, x, y, alpha, n_iter):
        self.x = x
        self.y = y
        self.m = self.x.shape[0]
        self.theta = np.zeros(self.m)
        self.alpha = alpha

    def predict(self, new_x):
        """Predict value of new_x"""
        pass

    def fit(self):
        """Execute gradient descent algorithm to estimate theta coefficients"""
        # Alther values of theta 
        # call cost_function() to caluclate cost
        costs = []
        for i in np.arange(n_iter):
            new_theta = self._gradient_descent()
            costs.append(self.cost_function(new_theta))

    def cost_function(self, theta):
        """Function to be minimized during regression"""
        # cost = (1 / 2m) * (np.dot(theta.T, x) - y) ** 2
        cost = (1 / 2 * self.m) * (np.dot(theta.T, self.x) - self.y) ** 2
        return cost

    def _gradient_descent(self):
        """Batch Gradient Descent equation to calculate theta"""
        # h_theta = np.dot(self.theta.T, self.x)
        # err = h_theta - self.y
        # del_j = np.dot(err, self.x)
        # theta_new = theta - ((alpha / m) * (del_j))

