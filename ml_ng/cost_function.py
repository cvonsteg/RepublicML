import numpy as np


class CostFunction:
    def __init__(self):

# J(theta) = 1/2m * sum(h_theta(x) - y) ** 2
# h_theta(x) = theta0 * x0 + theta1 * x1 + ... + thetaN * xN
# h_theta(x) = sigma.T * X
