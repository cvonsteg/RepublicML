import numpy as np
from republic_ml import supervised_models
x = np.array(range(1, 11))
y = np.array(range(2, 22, 2))
slr = supervised_models.Regression(x, y)
slr.fit()
Slope and Intercept calculated - model ready to use
f = np.array([12, 13, 14, 15])
slr.predict(f)
Model fit - y_hat prediction available
slr.y_hat
