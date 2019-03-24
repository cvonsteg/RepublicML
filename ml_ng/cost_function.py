import numpy as np


class CostFunction:
    def __init__(self) -> None:


# J(theta) = 1/2m * sum(h_theta(x) - y) ** 2
# h_theta(x) = theta0 * x0 + theta1 * x1 + ... + thetaN * xN
# h_theta(x) = sigma.T * X

s = np.array([1, 2, 3, 4])
r = np.array([5, 6, 7, 8])

s * r

np.dot(s, r.T)
