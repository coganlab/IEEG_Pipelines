import numpy as np
import pytest
import os


@pytest.mark.parametrize("mat, mask, axis, expected", [
    # Test case 1: No mask, axis=0
    (np.array([[1, 2, 3], [4, 5, 6]]), None, 0, ([2.5, 3.5, 4.5],
                                                 [1.5, 1.5, 1.5])),

    # Test case 2: No mask, axis=1
    (np.array([[1, 2, 3], [4, 5, 6]]), None, 1, ([2.0, 5.0], [1.0, 1.0])),

    # Test case 3: With mask, axis=0
    (np.array([[1, 2, 3], [4, np.nan, 6]]), np.array([[1, 1, 1], [1, 0, 1]]),
     0, ([2.5, 2.0, 4.5], [1.5, 1.0, 1.5])),

    # Test case 4: With mask, axis=1
    (np.array([[1, 2, 3], [4, np.nan, 6]]), np.array([[1, 1, 1], [1, 0, 1]]),
     1, ([2.0, 5.0], [1.0, 1.0])),
])
def test_dist(mat, mask, axis, expected):
    from ieeg.calc.stats import dist  # Import your actual module here

    result = dist(mat, mask, axis)
    assert np.allclose(result[0], expected[0])  # Check mean
    assert np.allclose(result[1], expected[1])  # Check standard deviation

@pytest.mark.parametrize("input_mat, shape, expected", [
    (np.zeros((10, 52)), (5, 104), np.zeros((5, 104))),
    (np.zeros((10, 50, 52)), (5, 50, 104), np.zeros((5, 50, 104))),
    (np.zeros((10, 50, 50)), (5, 50, 104), np.zeros((4, 50, 104))),
    (np.zeros((10, 100, 50, 52)), (5, 100, 50, 104),
     np.zeros((5, 100, 50, 104))),
    (np.zeros((10, 100, 50, 50)), (5, 100, 50, 104),
     np.zeros((4, 100, 50, 104))),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (1, 2, 4),
     np.array([[[1, 2, 5, 6], [3, 4, 7, 8]]]))
])
def test_same(input_mat, shape, expected):
    from ieeg.calc.stats import make_data_shape
    new_shape = make_data_shape(input_mat, shape)
    assert np.all(new_shape == expected)

