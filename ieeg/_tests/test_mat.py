import numpy as np
import pytest


@pytest.mark.parametrize("arrays, axis, expected_output", [
    # Test case 1: Concatenate along axis 0
    ([np.array([]), np.array([[1, 2], [3, 4]]),
      np.array([[5, 6, 7], [8, 9, 10]])],
     0,
     np.array([[1, 2, np.nan], [3, 4, np.nan], [5, 6, 7], [8, 9, 10]])),

    # Test case 2: Concatenate along axis 1
    ([np.ones((2, 1)), np.zeros((3, 1))], 1,
     np.array([[1, 0], [1, 0], [np.nan, 0]])),

    # Test case 3: Empty input arrays
    ([np.array([]), np.array([])], 0, None),

    # Test case 4: Concatenate along axis 2
    ([np.array([[[1]], [[2]]]), np.array([[[3], [4]], [[5], [6]]])],
     2,
     np.array([[[1, 3], [np.nan, 4]], [[ 2, 5],[np.nan, 6]]])),

    # Test case 5: Concatenate along axis 0 with empty array in the middle
    ([np.array([[1, 2], [3, 4]]), np.array([]),
      np.array([[5, 6, 7], [8, 9, 10]])],
     0,
     np.array([[1, 2, np.nan], [3, 4, np.nan], [5, 6, 7], [8, 9, 10]])),

    # Test case 6: Concatenate along axis 0 with empty arrays at the beginning
    # and end
    ([np.array([]), np.array([[1, 2], [3, 4]]), np.array([])],
     0,
     np.array([[1, 2], [3, 4]])),

    # Test case 7: Concatenate along axis -1 (last axis)
    ([np.array([[[1]], [[2]]]), np.array([[[3], [4]], [[5], [6]]])],
        -1,
        np.array([[[1, 3], [np.nan, 4]], [[2, 5], [np.nan, 6]]])),

    # Test case 8: Concatenate an array containing only nan values
    ([np.array([[np.nan, np.nan], [np.nan, np.nan]]),
      np.array([[5, 6, 7], [8, 9, 10]])],
     0,
     np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
               [5, 6, 7], [8, 9, 10]])),
])
def test_concatenate_arrays(arrays, axis, expected_output):
    from ieeg.calc.mat import concatenate_arrays, get_homogeneous_shapes
    print(f"Shapes {[arr.shape for arr in arrays]}")
    try:
        new = concatenate_arrays(arrays, axis)
        print(f"New shape {new.shape}")
        while axis < 0:
            axis += new.ndim
        congruency = new.shape == np.max(get_homogeneous_shapes(arrays),
                                         axis=0)
        print(congruency)
        assert all([con for i, con in enumerate(congruency) if i != axis])
        assert np.array_equal(new, expected_output, True)
    except ValueError as e:
        try:
            assert expected_output is None
        except AssertionError:
            raise e
