import numpy as np


class SupervisedModel:
    def __init__(self, x, y, alpha) -> None:
        self.x = np.matrix([np.ones(len(x)), x]).T
        self.y = np.matrix(y)
        self.m = self.x.shape[0]
        self.theta = np.matrix(np.zeros(self.x.shape[1]))
        self.alpha = alpha

    def predict(self, unknown_var):
        pass

    def fit(self):
        pass

    def hypothesis(self):
        pass

    def cost_function(self):
        pass

    def _gradient_descent_step(self):
        pass
