import numpy as np
from republic_ml import supervised_models 

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

slr = supervised_models.Regression(x, y)

slr.coefficients()
print(slr._y_intercept())
print(slr._slope())
print(f'y_hat for input 11 = {slr.y_hat(11)}')
