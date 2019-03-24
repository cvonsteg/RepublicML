import numpy as np
from ml_ng.linear_regression import Regression
from ml_ng.supervised_model import SupervisedModel
x = np.random.randint(low=0, high=5, size=10)
y = np.matrix(np.random.randint(low=5, high=10, size=10)).T

r = Regression(x=x, y=y, alpha=0.01, n_iter=1000)

r.x

r.y

r.theta

r.hypothesis(r.x)

r.predict([4, 5, 6, 7])

r.cost()
r.fit()
r.cost_dict
r.cost_dict[min(r.cost_dict)]

err = r.hypothesis(r.x) - r.y
err
np.dot(err.T, r.x)


r.theta
{1: r.theta}
_x = np.matrix([ones, x]).T
theta1 = np.random.randint(low=0, high=5, size=10)
theta2 = np.random.randint(low=0, high=5, size=10)
theta = np.matrix([5, 3])
theta = np.zeros(_x.shape[1])
# Given input X, we want to predict output y
# General form (hypothesis): h(x) = theta0 + x*theta1
# Determine theta




theta
_x
y
theta

def h_theta(X: np.matrix, theta: np.matrix) -> np.array:
    """
    Hypothesis function for machine learning.  Returns a predicted value of y
    (also known as y_hat)
    Parameters:
    - X:
        An np.matrix() object of model input parameters.

    General form of h(x): theta * X.T
    """
    return np.dot(theta, X.T).T




preddy = h_theta(_x, theta)
preddy
err = preddy - y
err.shape
_x.shape
err
_x
np.dot(err.T, _x)
theta - (3 * np.dot(err.T, _x))

a = {}
b = ('a', 1)
a[b[0]] = b[1]
a
