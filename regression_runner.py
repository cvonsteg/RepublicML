#! /usr/bin/env python3

import numpy as np
from republic_ml.supervised_models import LinearRegression

x = np.random.randint(low=0, high=20, size=100)
y = [i * np.random.rand() for i in x]
alpha = 0.001

reg = LinearRegression(x, y, alpha)
print(reg.theta)
reg.fit(n_iter=1000, plot=True)
print(reg.theta)
