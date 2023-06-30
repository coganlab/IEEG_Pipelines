import numpy as np



class LabeledArray(np.ndarray):
    """ A numpy array with labeled dimensions, acting like a dictionary.

    A numpy array with labeled dimensions. This class is useful for storing
    data that is not easily represented in a tabular format. It acts as a
    nested dictionary but its values map to elements of a stored numpy array.
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

        # Remove empty dimensions

        arr = inner_array(data)
        keys = inner_all_keys(data)
        return cls(arr, keys)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)

    def dropna(self) -> 'LabeledArray':
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

    def __getitem__(self, *keys):
        dim, keys = zip(*self._str_parse(*keys))
        out = super(LabeledArray, self).__getitem__(*keys)
        if isinstance(out, np.ndarray):
            new_labels = list(self.labels)
            for d in set(dim):
                new_labels.pop(d)
            setattr(out, 'labels', tuple(new_labels))
            # out = out.dropna()
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
        return str(self.to_dict())

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

    def combine_dims(self, dims: tuple[int, ...],
                     delim: str = '-') -> 'LabeledArray':
        """Combine the given dimensions into a single dimension.
        Parameters
        ----------
        dims : tuple[int, ...]
            The dimensions to combine.
        delim : str, optional
            The delimiter to use between the dimension labels, by default '-'.
        Returns
        -------
        LabeledArray
            The new LabeledArray with the combined dimensions.
        """

        # Create a new LabeledArray with the same data as the current instance
        new = self.copy()

        # Sort the dimensions in descending order to avoid changing the indices of
        # dimensions that have not been processed yet
        dims = sorted(dims, reverse=True)

        # Iterate over the dimensions to combine
        for dim in dims:
            # Create a new list of labels for the current dimension by joining the labels
            # of the current and next dimensions with the specified delimiter
            new_labels = [f'{l}{delim}{ll}' for l in new.labels[dim] for ll in
                          new.labels[dim + 1]]

            # Update the labels attribute of the new LabeledArray by replacing the labels
            # of the current and next dimensions with the new list of labels
            labels = list(new.labels)
            labels[dim] = new_labels
            labels.pop(dim + 1)
            new.labels = tuple(labels)

            # Update the shape attribute of the new LabeledArray by multiplying the sizes
            # of the current and next dimensions and removing the size of the next dimension
            shape = list(new.shape)
            shape[dim] *= shape[dim + 1]
            shape.pop(dim + 1)
            new.shape = tuple(shape)

            # Update the data attribute of the new LabeledArray by combining the data along
            # the current and next dimensions using np.reshape
            data = np.reshape(new,
                              new.shape[:dim] + (-1,) + new.shape[dim + 2:])
            new = LabeledArray(data, labels=new.labels)

        return new



def inner_all_keys(data: dict, keys: list = None, lvl: int = 0):
    if keys is None:
        keys = []
    if isinstance(data, (int, float, str, bool)) or np.isscalar(data):
        return
    elif isinstance(data, dict):
        if len(keys) < lvl + 1:
            keys.append(list(data.keys()))
        else:
            keys[lvl] += [k for k in data.keys() if k not in keys[lvl]]
        for d in data.values():
            inner_all_keys(d, keys, lvl+1)
    elif isinstance(data, np.ndarray):
        rows = range(data.shape[0])
        if len(keys) < lvl+1:
            keys.append(list(rows))
        else:
            keys[lvl] += [k for k in rows if k not in keys[lvl]]
        if len(data.shape) > 1:
            inner_all_keys(data[0], keys, lvl+1)
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")
    return tuple(map(tuple, keys))


def inner_array(data: dict | np.ndarray) -> np.ndarray | None:
    if len(data) == 0:
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
    if len(data) == 0:
        return
    elif isinstance(data, LabeledArray):
        return {i: inner_dict(d) for i, d in enumerate(data)}
    else:
        return data

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

    data = LabeledArray.from_dict(dict(
        power=load_dict(layout, conds, "power", False),
        zscore=load_dict(layout, conds, "zscore", False)))

    power = data["power"]