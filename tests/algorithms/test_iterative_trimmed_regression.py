import numpy as np
import unittest

from robust_linear_regression.algorithms.iterative_trimmed_regression import IterativeTrimmedRegression
import robust_linear_regression.utils.constants as constants


class TestChangeConditionNumber(unittest.TestCase):
    """A unit test for IterativeTrimmedRegression.
    """
    def test_get_signal_estimation(self):
        """
        """
        H = [[1, 2], [2, 3], [1, 3]]
        y = [9, 9, 9]
        x = [1, 2]
        x_updated = IterativeTrimmedRegression().get_signal_estimation(y, H, x, 2)
        x_expect = [0, 3]
        np.testing.assert_array_almost_equal(x_expect, x_updated, decimal=7)
