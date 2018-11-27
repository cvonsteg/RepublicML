from unittest import TestCase
import numpy as np
from main.utils.vector import Vector

class TestBaseClass(TestCase):
    def setUp(self) -> None:
        self.v1 = Vector([1, 2, 3, 4, 5])

    def test_mean(self):
        self.assertEqual(self.v1.mean, 3) 

    def test_median(self):
        self.assertEqual(self.v1.median, 3)

    def test_len(self):
        self.assertEqual(len(self.v1), 55 ** 0.5)
