import numpy as np
cimport numpy as cnp
cimport cython
import concat

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict inner_dict(data):
    """Convert a nested array to a nested dictionary.

    Parameters
    ----------
    data : np.ndarray
        The nested array to convert.

    Returns
    -------
    dict or None
        The converted nested dictionary.

    Examples
    --------
    >>> data = np.array([[[1]]])
    >>> inner_dict(data) # doctest: +ELLIPSIS +SKIP
    {0: {0: {0: 1}}}
    >>> data = np.array([[[1, np.nan]],
    ...                  [[2, 3]]])
    >>> inner_dict(data) # doctest: +ELLIPSIS +SKIP
    {0: {0: {0: ...
    """
    cdef dict result = {}
    cdef Py_ssize_t i
    cdef object d

    if isinstance(data, dict):
        return data
    elif hasattr(data, 'ndim'):
        for i, d in enumerate(data):
            if data.ndim == 1:
                result[i] = d
            elif len(d) > 0:
                result[i] = inner_dict(d)
            else:
                continue
        return result
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray inner_array(data):
    """Convert a nested dictionary to a nested array.

    Parameters
    ----------
    data : dict or np.ndarray
        The nested dictionary to convert.

    Returns
    -------
    np.ndarray or None
        The converted nested array.

    Examples
    --------
    >>> data = {'a': {'b': {'c': 1}}}
    >>> inner_array(data)
    array([[[1]]])
    >>> data = {'a': {'b': {'c': 1}}, 'd': {'b': {'c': 2, 'e': 3}}}
    >>> inner_array(data)
    array([[[ 1., nan]],
    <BLANKLINE>
           [[ 2.,  3.]]])
    """
    cdef object gen_arr, arr

    if np.isscalar(data):
        return data
    elif isinstance(data, dict):
        gen_arr = (inner_array(d) for d in data.values())
        arr = [a for a in gen_arr if a is not None]
        if len(arr) > 0:
            return concat.concatenate_arrays(arr, axis=None)

    # Call np.atleast_1d once and store the result in a variable
    data_1d = np.atleast_1d(data)

    # Use the stored result to check the length of data
    if len(data_1d) == 0:
        return
    elif len(data_1d) == 1:
        return data
    else:
        return np.array(data)


cpdef void add_to_list_if_not_present(list lst, element):
    """Add an element to a list if it is not present. Runs in O(1) time.

    Parameters
    ----------
    lst : list
        The list to add the element to.
    element : Iterable
        The element to add to the list.

    References
    ----------
    [1] https://www.youtube.com/watch?v=PXWL_Xzyrp4

    Examples
    --------
    >>> lst = [1, 2, 3]
    >>> add_to_list_if_not_present(lst, [3, 4, 5])
    >>> lst
    [1, 2, 3, 4, 5]
    """
    seen = set(lst)
    lst.extend(x for x in element if not (x in seen or seen.add(x)))


cpdef tuple[tuple[str]] inner_all_keys(dict data, list keys = None, int lvl = 0):
    """Get all keys of a nested dictionary.

    Parameters
    ----------
    data : dict
        The nested dictionary to get the keys of.
    keys : list, optional
        The list of keys, by default None
    lvl : int, optional
        The level of the dictionary, by default 0

    Returns
    -------
    tuple
        The tuple of keys.

    Examples
    --------
    >>> data = {'a': {'b': {'c': 1}}}
    >>> inner_all_keys(data)
    (('a',), ('b',), ('c',))
    >>> data = {'a': {'b': {'c': 1}}, 'd': {'b': {'c': 2, 'e': 3}}}
    >>> inner_all_keys(data)
    (('a', 'd'), ('b',), ('c', 'e'))
    """
    if keys is None:
        keys = []
    if isinstance(data, dict):
        if len(keys) < lvl + 1:
            keys.append(list(data.keys()))
        else:
            add_to_list_if_not_present(keys[lvl], data.keys())
        for d in data.values():
            if np.isscalar(d):
                continue
            inner_all_keys(d, keys, lvl+1)
    elif isinstance(data, np.ndarray):
        data = np.atleast_1d(data)
        rows = range(data.shape[0])
        if len(keys) < lvl+1:
            keys.append(list(rows))
        else:
            add_to_list_if_not_present(keys[lvl], rows)
        if len(data.shape) > 1:
            if not np.isscalar(data[0]):
                inner_all_keys(data[0], keys, lvl+1)
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")
    return tuple(map(tuple, keys))
