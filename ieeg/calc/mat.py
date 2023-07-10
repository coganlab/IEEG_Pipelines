import numpy as np
from collections.abc import Iterable

def iter_nest_dict(d: dict, _lvl: int = 0, _coords=()):
    """Iterate over a nested dictionary, yielding the key and value.

    Parameters
    ----------
    d : dict
        The dictionary to iterate over.
    _lvl : int, optional
        The current level of nesting, by default 0
    _coords : tuple, optional
        The current coordinates of the array, by default ()

    Yields
    ------
    tuple
        The key and value of the dictionary.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            yield from iter_nest_dict(v, _lvl + 1, _coords + (k,))
        elif isinstance(v, np.ndarray):
            yield from iter_nest_dict({i: val for i, val in enumerate(v)
                                       }, _lvl + 1, _coords + (k,))
        else:
            yield _coords + (k,), v


class LabeledArray(np.ndarray):
    """ A numpy array with labeled dimensions, acting like a dictionary.

    A numpy array with labeled dimensions. This class is useful for storing
    data that is not easily represented in a tabular format. It acts as a
    nested dictionary but its values map to elements of a stored numpy array.

    Parameters
    ----------
    input_array : array_like
        The array to store in the LabeledArray.
    labels : tuple[tuple[str, ...], ...], optional
        The labels for each dimension of the array, by default ().
    **kwargs
        Additional arguments to pass to np.asarray.

    Attributes
    ----------
    labels : tuple[tuple[str, ...], ...]
        The labels for each dimension of the array.
    array : np.ndarray
        The array stored in the LabeledArray.

    Examples
    --------
    >>> import numpy as np
    >>> from ieeg.calc.mat import LabeledArray
    >>> arr = np.ones((2, 3, 4))
    >>> labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
    >>> la = LabeledArray(arr, labels)
    >>> la.to_dict()
    {'a': {'c': {'f': 1.0, 'g': 1.0, 'h': 1.0, 'i': 1.0},
                 'd': {'f': 1.0, 'g': 1.0, 'h': 1.0, 'i': 1.0},
                'e': {'f': 1.0, 'g': 1.0, 'h': 1.0, 'i': 1.0}},
        'b': {'c': {'f': 1.0, 'g': 1.0, 'h': 1.0, 'i': 1.0},
                    'd': {'f': 1.0, 'g': 1.0, 'h': 1.0, 'i': 1.0},
                    'e': {'f': 1.0, 'g': 1.0, 'h': 1.0, 'i': 1.0}}}
    >>> la['a', 'c', 'f'] = 2.
    >>> la['a', 'c', 'f']
    2.0

    References
    ----------
    [1] https://numpy.org/doc/stable/user/basics.subclassing.html
    """

    def __new__(cls, input_array, labels: tuple[tuple[str, ...], ...] = (),
                **kwargs):
        obj = np.asarray(input_array, **kwargs).view(cls)
        labels = list(labels)
        for i in range(obj.ndim):
            if len(labels) < i + 1:
                labels.append(tuple(range(obj.shape[i])))
        obj.labels = tuple(labels)
        return obj

    @classmethod
    def from_dict(cls, data: dict) -> 'LabeledArray':
        """Create a LabeledArray from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary to convert to a LabeledArray.

        Returns
        -------
        LabeledArray
            The LabeledArray created from the dictionary.
        """

        arr = inner_array(data)
        keys = inner_all_keys(data)
        return cls(arr, keys)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)

    def dropna(self) -> 'LabeledArray':
        """Remove all nan values from the array."""
        new_labels = list(self.labels)
        idx = []
        for i in range(self.ndim):
            axes = tuple(j for j in range(self.ndim) if j != i)
            mask = np.all(np.isnan(np.array(self)), axis=axes)
            if np.any(mask):
                new_labels[i] = tuple(np.array(new_labels[i])[~mask])
            idx.append(~mask)
        index = np.ix_(*idx)
        new_array = LabeledArray(np.array(self)[index], new_labels)
        return new_array

    @property
    def label_map(self) -> tuple[dict[str: int, ...], ...]:
        """maps the labels to the indices of the array."""
        return tuple({l: i for i, l in enumerate(labels)}
                     for labels in self.labels)

    def _str_parse(self, *keys) -> tuple[int, int]:
        for key in keys:
            match key:
                case list() | tuple():
                    key = list(key)
                    while key:
                        for value in self._str_parse(key.pop(0)):
                            yield value
                case str():
                    i = 0
                    while key not in self.labels[i]:
                        i += 1
                        if i > self.ndim:
                            raise KeyError(f'{key} not found in labels')
                    key = self.label_map[i][key]
                    yield i, key
                case _:
                    yield 0, key

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = []
        new_labels = []
        dim = 0
        for key in keys:
            if isinstance(key, str):
                i, key = next(self._str_parse(key))
                new_keys.append(key)
                while dim < i:
                    new_labels.append(self.labels[dim])
                    dim += 1
                dim += 1
            elif key is Ellipsis:
                new_keys.append(key)
                num_ellipsis_dims = self.ndim - len(keys) + 1
                new_labels.extend(self.labels[dim:dim + num_ellipsis_dims])
                dim += num_ellipsis_dims
            elif key is None:
                new_keys.append(key)
                new_labels.append(self.labels[dim])
                dim += 1
            elif isinstance(key, int):
                new_keys.append(key)
                dim += 1
            elif isinstance(key, slice):
                new_keys.append(key)
                new_labels.append(self.labels[dim])
                dim += 1
            else:
                new_keys.append(key)
                new_labels.append(self.labels[dim])
                dim += 1
        while dim < self.ndim:
            new_labels.append(self.labels[dim])
            dim += 1
        out = super().__getitem__(tuple(new_keys))
        if isinstance(out, np.ndarray):
            setattr(out, 'labels', tuple(new_labels))
        return out

    def __setitem__(self, key, value):
        dim, num_key = self._str_parse(key)
        if key not in self.labels[dim]:
            self.labels[dim] += (key,)
        super(LabeledArray, self).__setitem__(num_key, value)

    def __delitem__(self, key):
        dim, num_key = self._str_parse(key)
        if key in self.labels[dim]:
            self.labels = list(self.labels)
            self.labels[dim] = tuple(l for l in self.labels[dim]
                                     if l != key)
            self.labels = tuple(self.labels)
        super(LabeledArray, self).__delitem__(key)

    def __repr__(self):
        """Display like a dictionary with labels as keys"""
        size = self.nbytes
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if size < 1024.0 or unit == 'PiB':
                break
            size /= 1024.0

        return f'{super().__repr__()}, labels={self.labels} ~{size:.2f} {unit}'

    def __eq__(self, other):
        if isinstance(other, LabeledArray):
            return np.array_equal(self, other) and self.labels == other.labels
        return super().__eq__(other)

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        out = {}
        for k, v in self.items():
            if isinstance(v, LabeledArray):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def items(self):
        return zip(self.keys(), self.values())

    def keys(self):
        return (l for l in self.labels[0])

    def values(self):
        return (a for a in self)

    def combine(self, levels: tuple[int, int],
                delim: str = '-') -> 'LabeledArray':
        """Combine any levels of a LabeledArray into the lower level

        Takes the input LabeledArray and rearranges its dimensions.

        Parameters
        ----------
        levels : tuple[int, int]
            The levels to combine, e.g. (0, 1) will combine the 1st and 2nd level
            of the array labels into one level at the 2nd level.
        delim : str, optional
            The delimiter to use when combining labels, by default '-'

        Returns
        -------
        LabeledArray
            The combined LabeledArray

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1}}}
        >>> ad = LabeledArray.from_dict(data)
        >>> ad.combine((0, 2))
        LabeledArray([1], labels=(('b',), ('a-c',)))
        """

        assert levels[0] >= 0, "first level must be >= 0"
        assert levels[1] > levels[0], "second level must be > first level"

        new_labels = list(self.labels)
        new_labels.pop(levels[0])
        new_labels[levels[1] - 1] = tuple(
            f'{self.labels[levels[0]][i]}{delim}{l}' for i in
            range(self.shape[levels[0]]) for l in self.labels[levels[1]])

        new_array = np.moveaxis(self, (levels[0], levels[1]), (0, 1)).reshape(
            self.shape[levels[1]], -1)

        return LabeledArray(new_array, new_labels)


def add_to_list_if_not_present(lst: list, element: Iterable):
    """Add an element to a list if it is not present. Runs in O(1) time."""
    seen = set(lst)
    lst.extend(x for x in element if not (x in seen or seen.add(x)))


def inner_all_keys(data: dict, keys: list = None, lvl: int = 0):
    """Get all keys of a nested dictionary."""
    if keys is None:
        keys = []
    if np.isscalar(data):
        return
    elif isinstance(data, dict):
        if len(keys) < lvl + 1:
            keys.append(list(data.keys()))
        else:
            add_to_list_if_not_present(keys[lvl], data.keys())
        for d in data.values():
            inner_all_keys(d, keys, lvl+1)
    elif isinstance(data, np.ndarray):
        rows = range(data.shape[0])
        if len(keys) < lvl+1:
            keys.append(list(rows))
        else:
            add_to_list_if_not_present(keys[lvl], rows)
        if len(data.shape) > 1:
            inner_all_keys(data[0], keys, lvl+1)
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")
    return tuple(map(tuple, keys))


def inner_array(data: dict | np.ndarray) -> np.ndarray | None:
    """Convert a nested dictionary to a nested array."""
    if np.isscalar(data):
        return data
    elif len(data) == 0:
        return
    elif isinstance(data, dict):
        arr = (inner_array(d) for d in data.values())
        arr = [a for a in arr if a is not None]
        if len(arr) > 0:
            return concatenate_arrays(arr, axis=None)
    else:
        return np.array(data)


def inner_dict(data: np.ndarray) -> dict | None:
    """Convert a nested array to a nested dictionary."""
    if np.isscalar(data):
        return data
    elif len(data) == 0:
        return
    elif isinstance(data, np.ndarray):
        return {i: inner_dict(d) for i, d in enumerate(data)}
    else:
        return data


def combine(data: dict, levels: tuple[int, int], delim: str = '-') -> dict:
    """Combine any levels of a nested dict into the lower level

    Takes the input nested dict and rearranges the top and bottom
    sub-dictionary.

    Parameters
    data: dict
        The nested dict to combine
    levels: tuple[int, int]
        The levels to combine, e.g. (0, 1) will combine the 1st and 2nd level
        of the dict keys into one level at the 2nd level.

    Returns
    dict
        The combined dict

    Examples
    >>> data = {'a': {'b': {'c': 1}}}
    >>> combine(data, (0, 2))
    {'b': {'a-c': 1}}

    >>> data = {'a': {'b': {'c': 1}}, 'd': {'b': {'c': 2, 'e': 3}}}
    >>> combine(data, (0, 2))
    {'b': {'a-c': 1, 'd-c': 2, 'd-e': 3}}
    """

    assert levels[0] >= 0, "first level must be >= 0"
    assert levels[1] > levels[0], "second level must be > first level"

    def _combine_helper(data, levels, depth, keys):
        if depth == levels[1]:
            return {f'{keys[levels[0]]}{delim}{k}': v for k, v in data.items()}
        elif depth == levels[0]:
            new_dict = {}
            for k, v in data.items():
                for k2, v2 in _combine_helper(v, levels, depth + 1,
                                             keys + [k]).items():
                    if isinstance(v2, dict):
                        if k2 in new_dict:
                            new_dict[k2] = _merge(new_dict[k2], v2)
                        else:
                            new_dict[k2] = v2
                    else:
                        new_dict[k2] = v2
            return new_dict
        else:
            return {k: _combine_helper(v, levels, depth + 1, keys + [k]) for
                    k, v in data.items()}

    def _merge(d1: dict, d2: dict) -> dict:
        for k, v in d2.items():
            if isinstance(v, dict):
                d1[k] = _merge(d1.get(k, {}), v)
            else:
                d1[k] = v
        return d1

    result = _combine_helper(data, levels, 0, [])

    return result


def concatenate_arrays(arrays: list[np.ndarray], axis: int = None
                       ) -> np.ndarray:
    """Concatenate arrays along a specified axis, filling in empty arrays with
    nan values.

    Parameters
    ----------
    arrays
        A list of arrays to concatenate
    axis
        The axis along which to concatenate the arrays

    Returns
    -------
    result
        The concatenated arrays
    """

    if axis is None:
        axis = 0
        arrays = [np.expand_dims(ar, axis) for ar in arrays]

    while axis < 0:
        axis += max(ar.ndim for ar in arrays)

    # Determine the maximum shape along the specified axis
    max_shape = np.max(get_homogeneous_shapes(arrays), axis=0)

    # Create a list to store the modified arrays
    modified_arrays = []

    # Iterate over the arrays
    for arr in arrays:
        if len(arr) == 0:
            continue
        # Determine the shape of the array
        arr_shape = list(max_shape)
        arr_shape[axis] = arr.shape[axis]

        # Create an array filled with nan values
        nan_array = np.full(arr_shape, np.nan)

        # Fill in the array with the original values
        indexing = [slice(None)] * arr.ndim
        for ax in range(arr.ndim):
            if ax == axis:
                continue
            indexing[ax] = slice(0, arr.shape[ax])
        nan_array[tuple(indexing)] = arr

        # Append the modified array to the list
        modified_arrays.append(nan_array)

    # Concatenate the modified arrays along the specified axis
    result = np.concatenate(modified_arrays, axis=axis)

    return result


def get_homogeneous_shapes(arrays):
    # Determine the maximum number of dimensions among the input arrays
    max_dims = max([arr.ndim for arr in arrays])

    # Create a list to store the shapes with a homogeneous number of dimensions
    homogeneous_shapes = []

    # Iterate over the arrays
    for arr in arrays:
        # Get the shape of the array
        # Handle the case of an empty array
        if len(arr) == 0:
            shape = (0,)
            dims = 1
        else:
            shape = arr.shape
            dims = arr.ndim

        # Pad the shape tuple with additional dimensions if necessary
        num_dims_to_pad = max_dims - dims
        shape += (1,) * num_dims_to_pad

        # Add the shape to the list
        homogeneous_shapes.append(shape)

    return homogeneous_shapes


def get_elbow(data: np.ndarray) -> int:
    """Draws a line between the first and last points in a dataset and finds
    the point furthest from that line.

    Parameters
    ----------
    data : array
        The data to find the elbow in.

    Returns
    -------
    int
        The index of the elbow point.
    """
    nPoints = len(data)
    allCoord = np.vstack((range(nPoints), data)).T
    np.array([range(nPoints), data])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(
        lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    # set distance to points below lineVec to 0
    distToLine[vecToLine[:, 1] < 0] = 0
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def stitch_mats(mats: list[np.ndarray], overlaps: list[int], axis: int = 0
                ) -> np.ndarray:
    """break up the matrices into their overlapping and non-overlapping parts
    then stitch them back together

    Parameters
    ----------
    mats : list
        The matrices to stitch together
    overlaps : list
        The number of overlapping rows between each matrix
    axis : int, optional
        The axis to stitch along, by default 0

    Returns
    -------
    np.ndarray
        The stitched matrix
    """
    stitches = [mats[0]]
    if len(mats) != len(overlaps) + 1:
        raise ValueError("The number of matrices must be one more than the num"
                         "ber of overlaps")
    for i, over in enumerate(overlaps):
        stitches = stitches[:-2] + merge(stitches[-1], mats[i+1], over, axis)
    return np.concatenate(stitches, axis=axis)


def merge(mat1: np.ndarray, mat2: np.ndarray, overlap: int, axis: int = 0
          ) -> list[np.ndarray]:
    """Take two arrays and merge them over the overlap gradually"""
    sl = [slice(None)] * mat1.ndim
    sl[axis] = slice(0, mat1.shape[axis]-overlap)
    start = mat1[tuple(sl)]
    sl[axis] = slice(mat1.shape[axis]-overlap, mat1.shape[axis])
    middle1 = np.multiply(np.linspace(1, 0, overlap), mat1[tuple(sl)])
    sl[axis] = slice(0, overlap)
    middle2 = np.multiply(np.linspace(0, 1, overlap), mat2[tuple(sl)])
    middle = np.add(middle1, middle2)
    sl[axis] = slice(overlap, mat2.shape[axis])
    last = mat2[tuple(sl)]
    return [start, middle, last]


if __name__ == "__main__":
    import os
    from ieeg.io import get_data
    from utils.mat_load import load_dict
    import mne
    ins, axis, exp = ([np.array([]), np.array([[1., 2.], [3., 4.]]),
                       np.array([[5., 6., 7.], [8., 9., 10.]])], 0,
                       np.array([[1, 2, np.nan], [3, 4, np.nan],
                                 [5, 6, 7], [8, 9, 10]]))
    outs = concatenate_arrays(ins, axis)
    ar = LabeledArray.from_dict(dict(a=ins[1], b=ins[2]))
    x = ar["a"]
    y = ar.to_dict()
    conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
             "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
             "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
             "go_jl": (-0.5, 1.5)}
    task = "SentenceRep"
    root = os.path.expanduser("~/Box/CoganLab")
    layout = get_data(task, root=root)

    mne.set_log_level("ERROR")

    # data = LabeledArray.from_dict(dict(
    #     power=load_dict(layout, conds, "power", False),
    #     zscore=load_dict(layout, conds, "zscore", False)))

    dict_data = dict(
        power=load_dict(layout, conds, "power", False))
        # zscore=load_dict(layout, conds, "zscore", False))

    keys = inner_all_keys(dict_data)

    data = LabeledArray.from_dict(dict_data)
    data.__repr__()

    # data = SparseArray(dict_data)

    # power = data["power"]