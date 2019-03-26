#! /usr/bin/env python3

import numpy as np
from republic_ml.supervised_models import LinearRegression

x = np.random.randint(low=0, high=20, size=100)
y = (x * np.random.rand()) + np.random.rand()
alpha = 0.001

reg = LinearRegression(x, y, alpha)
print(reg.theta)
reg.fit(n_iter=100, plot=True)
print(reg.theta)
