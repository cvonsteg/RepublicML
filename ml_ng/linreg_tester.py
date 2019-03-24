#! /usr/bin/env python

import numpy as np
from linreg import Regression


x = np.array([1, 2, 3, 4])
y = np.array([9, 11, 13, 15])
alpha = 0.3
n_iter = 1000

r = Regression(x, y, alpha, n_iter)

print(f'x = {r.x}')
print(f' y  = {r.y}')
print(f'theta = {r.theta}')
