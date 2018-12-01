import numpy as np
from republic_ml import *

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
slr = LinRegSimple(x, y)

slr.y_hat(np.array([10, 11, 12, 13]))
