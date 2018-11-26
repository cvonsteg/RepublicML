from unittest import TestCase
import numpy as np
from main.utils.base_model import BaseModel

class TestBaseClass(TestCase):
    def setUp(self) -> None:
        self.base_vector1 = np.array([1, 2, 3, 4, 5])
        self.base_vector2 = np.array([9, 8, 7, 6, 5])
        self.model = BaseModel(self.base_vector1, self.base_vector2)

    def test_mean(self):
        vector = np.array([1, 2, 3, 4, 5])
        value = self.model.mean(vector)
        self.assertEqual(value, (sum(vector) / len(vector)))

    def test_median(self):
        vector = np.array([1, 2, 3, 4, 5])
        value = self.model.median(vector)
        self.assertEqual(value, 3)
