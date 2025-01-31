"""Test function for SMOTE."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import TSMOTE
from aeon.utils.validation._dependencies import _check_soft_dependencies


def test_tsmote():
    """Test the SMOTE class.

    This function creates a 3D numpy array, applies
    SMOTE using the SMOTE class, and asserts that the
    transformed data has a balanced number of samples.
    """
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)
    transformer = TSMOTE()
    transformer.fit(X, y)
    res_X, res_y = transformer.transform(X, y)
    _, res_count = np.unique(res_y, return_counts=True)

    assert len(res_X) == 2 * majority_num
    assert len(res_y) == 2 * majority_num
    assert res_count[0] == majority_num
    assert res_count[1] == majority_num
