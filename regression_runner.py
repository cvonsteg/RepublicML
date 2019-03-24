#/usr/bin/env python3

import numpy as np
from republic_ml.supervised_models import LinearRegression

x = np.random.randint(low=0, high=20, size=100)
y = np.random.randint(low=10, high=100, size=100)
alpha = 0.01

reg = LinearRegression(x, y, alpha)
print(reg.theta)
reg.fit(n_iter=1000)
print(reg.theta)
