import numpy as np


x = np.random.randint(low=0, high=5, size=10)
ones = np.ones(len(x))
_x = np.matrix([ones, x])
y = np.random.randint(low=5, high=10, size=10)
theta1 = np.random.randint(low=0, high=5, size=10)
theta2 = np.random.randint(low=0, high=5, size=10)
theta = np.matrix([5, 3])

# Given input X, we want to predict output y
# General form (hypothesis): h(x) = theta0 + x*theta1
# Determine theta

theta
_x
X1 = np.array([1, 3])
theta = np.array([2, 4])


def h_theta(X: np.matrix, theta: np.matrix) -> np.array:
    """
    General form of h(x): sigma.T * X
    """
    return np.dot(theta, X)

h_theta(_x, theta)

a = {}
b = ('a', 1)
a[b[0]] = b[1]
a
