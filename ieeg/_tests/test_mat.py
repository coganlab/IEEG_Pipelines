import numpy as np
import pytest
from ieeg.calc.mat import concatenate_arrays, get_homogeneous_shapes, \
    LabeledArray, combine


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

    # Test case 9: Concatenate along new axis
    ([np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]])],
        None,
        np.array([[[1, 2, np.nan], [3, 4, np.nan]], [[5, 6, 7], [8, 9, 10]]]))
])
def test_concatenate_arrays(arrays, axis, expected_output):
    print(f"Shapes {[arr.shape for arr in arrays]}")
    try:
        new = concatenate_arrays(arrays, axis)
        print(f"New shape {new.shape}")
        if axis is None:
            axis = 0
            arrays = [np.expand_dims(arr, axis) for arr in arrays]
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


# Test creation of ArrayDict
def test_array_creation():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    ad = LabeledArray.from_dict(data)
    assert isinstance(ad, LabeledArray)


# Test conversion to numpy array
def test_array_to_array():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    ad = LabeledArray.from_dict(data)
    np_array = np.array([[[1, 2, 3], [4, 5, np.nan]]])
    assert np.array_equal(ad, np_array, True)


# Test getting all keys
def test_array_all_keys():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    ad = LabeledArray.from_dict(data)
    keys = (('a',), ('b', 'f'), ('c', 'd', 'e'))
    assert ad.labels == keys


# Test getting all keys in a really nested ArrayDict
def test_array_all_keys_nested():
    data = {'a': {'b': {'c': {'d': {'e': {'f': {'g': {'h': {'i': {'j': {
        'k': 1}}}}}}}}}}}
    ad = LabeledArray.from_dict(data)
    keys = (('a',), ('b',), ('c',), ('d',), ('e',), ('f',), ('g',), ('h',),
            ('i',), ('j',), ('k',))
    assert ad.labels == keys


# Test getting all keys in a large ArrayDict (10000 keys)
def test_array_all_keys_large():
    data = {str(i): i for i in range(1000000)}
    ad = LabeledArray.from_dict(data)
    keys = (tuple(map(str, range(1000000))),)
    assert ad.labels == keys


# Test indexing with a single key
def test_array_single_key_indexing():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    ad = LabeledArray.from_dict(data)
    subset = LabeledArray.from_dict(**{'c': 1, 'd': 2, 'e': 3})
    assert ad['a']['b'] == subset


# Test indexing with a tuple of keys that leads to a scalar value
def test_array_scalar_value_indexing():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    ad = LabeledArray.from_dict(data)
    assert ad['a']['b']['d'] == 2


# Test shape property
def test_array_shape():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    ad = LabeledArray.from_dict(data)
    assert ad.shape == (1, 2, 3)


# Test combine dimensions
def test_combine_dimensions():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    new = combine(data, (0, 2))
    assert new == {'a': {'b-c': 1, 'b-d': 2, 'b-e': 3, 'f-c': 4, 'f-d': 5}}


# Test combine dimensions with non contiguous dimensions
def test_array_dict_combine_dimensions_non_contiguous():
    data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'd': 5}}}
    new = combine(data, (0, 2))
    assert new == {'b': {'a-c': 1, 'a-d': 2, 'a-e': 3}, 'f': {'a-c': 4, 'a-d': 5}}


# Test combine dimensions with arrays
# def test_array_dict_combine_dimensions_with_arrays():
#     data = {'b': {'c': np.array([1, 2, 3]), 'd': np.array([1, 2, 3])},
#                   'f': {'c': np.array([1, 2, 3])}}
#     ad = LabeledArray.from_dict(**data)
#     new = ad.combine_dims((1, 2))
#     assert new['b'] == {'c-0': 1, 'c-1': 2, 'c-2': 3, 'd-0': 1, 'd-1': 2,
#                         'd-2': 3}

