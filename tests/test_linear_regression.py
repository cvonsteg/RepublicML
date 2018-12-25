import numpy as np
from unittest import TestCase

from republic_ml import supervised_models


class TestLinReg(TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([10, 20, 30, 40, 50])

        self.slr = supervised_models.Regression(self.x, self.y)

    def test_sum_of_squares(self):
        result = self.slr.sum_of_squares()
        sos_x = (1**2 + 2**2 + 3**2 + 4**2 + 5**2) ** 0.5
        sos_y = (10**2 + 20**2 + 30**2 + 40**2 + 50**2) ** 0.5
        self.assertEqual(result, (sos_x, sos_y))

    def test_slope(self):
        slope = self.slr._slope()
        self.assertEqual(slope, 10)

    def test_y_intercept(self):
        y_int = self.slr._y_intercept()
        self.assertEqual(y_int, 0)

    def test_predict(self):
        self.slr.fit()
        self.slr.predict(7)
        self.assertEqual(self.slr.y_hat, 70)

    def test_rmse(self):
        self.slr.fit()
        self.slr.predict(self.x)
        rmse = self.slr.rmse()
        self.assertEqual(rmse, 0)
