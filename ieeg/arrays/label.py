import functools
from collections.abc import Iterable

import mne
from ieeg.calc.fast import concatenate_arrays

import numpy as np
from numpy.typing import ArrayLike

import ieeg


def iter_nest_dict(d: dict, iter_arrays: bool = False) -> Iterable[tuple]:
    """Iterate over a nested dictionary, yielding the key and value.

    Parameters
    ----------
    d : dict
        The dictionary to iterate over.

    Yields
    ------
    tuple
        The key and value of the dictionary.

    Examples
    --------
    >>> d = {'a': {'b': 1, 'c': 2}, 'd': {'e': 3, 'f': 4}}
    >>> for k, v in iter_nest_dict(d):
    ...     print(k, v)
    ('a', 'b') 1
    ('a', 'c') 2
    ('d', 'e') 3
    ('d', 'f') 4
    >>> d = {'a': {'b': np.array([1, 2]), 'c': 2}, 'd': {'e': 3, 'f': 4}}
    >>> for k, v in iter_nest_dict(d, iter_arrays=False):
    ...     print(k, v)
    ('a', 'b') [1 2]
    ('a', 'c') 2
    ('d', 'e') 3
    ('d', 'f') 4
    >>> for k, v in iter_nest_dict(d, iter_arrays=True):
    ...     print(k, v)
    ('a', 'b', 0) 1
    ('a', 'b', 1) 2
    ('a', 'c') 2
    ('d', 'e') 3
    ('d', 'f') 4
    """
    stack = [(d, [])]
    if not iter_arrays:
        while stack:
            current, path = stack.pop()
            if isinstance(current, dict):
                # Reverse to maintain order
                for k, v in reversed(current.items()):
                    stack.append((v, path + [k]))
            else:
                yield tuple(path), current
    else:
        while stack:
            current, path = stack.pop()
            if isinstance(current, dict):
                for k, v in reversed(current.items()):
                    stack.append((v, path + [k]))
            elif isinstance(current, np.ndarray):
                for i, val in reversed(list(enumerate(current))):
                    stack.append((val, path + [i]))
            else:
                yield tuple(path), current


def lcs(*strings: str) -> str:
    """Find the longest common substring in a list of strings.

    Parameters
    ----------
    *strings : str
        The strings to find the longest common substring of.

    Returns
    -------
    str
        The longest common substring in the list of strings.

    Examples
    --------
    >>> lcs('ABAB')
    'ABAB'
    >>> lcs('ABAB', 'BABA')
    'ABA'
    >>> lcs('ABAB', 'BABA', 'ABBA')
    'AB'
    """
    if not strings:
        return ""

    def _lcs_two_strings(s1, s2):
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        max_len = 0
        end_pos = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        end_pos = i

        return s1[end_pos - max_len:end_pos]

    common_substr = strings[0]
    for string in strings[1:]:
        common_substr = _lcs_two_strings(common_substr, string)
        if not common_substr:
            break

    return common_substr


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
    delimiter : str, optional
        The delimiter to use when combining labels, by default '-'
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
    >>> np.set_printoptions(legacy='1.21')
    >>> from ieeg.arrays.label import LabeledArray
    >>> arr = np.ones((2, 3, 4), dtype=int)
    >>> labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
    >>> la = LabeledArray(arr, labels)
    >>> la
    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],
    <BLANKLINE>
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]])
    labels(['a', 'b']
           ['c', 'd', 'e']
           ['f', 'g', 'h', 'i'])
    >>> la.to_dict() # doctest: +ELLIPSIS
    {'a': {'c': {'f': 1, 'g': 1, 'h': 1, 'i': 1}, 'd': {'f': 1, 'g': 1,...
    >>> la['a', 'c', 'f'] = 2
    >>> la['a', 'c', 'f']
    2
    >>> la['a', 'c']
    array([2, 1, 1, 1])
    labels(['f', 'g', 'h', 'i'])
    >>> la['a'].labels
    [['c', 'd', 'e'], ['f', 'g', 'h', 'i']]
    >>> la['a','d'] = np.array([3,3,3,3])
    >>> la[('a','b'), :]
    array([[[2, 1, 1, 1],
            [3, 3, 3, 3],
            [1, 1, 1, 1]],
    <BLANKLINE>
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]])
    labels(['a', 'b']
           ['c', 'd', 'e']
           ['f', 'g', 'h', 'i'])
    >>> la[np.array([False, True]),]
    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]])
    labels(['b']
           ['c', 'd', 'e']
           ['f', 'g', 'h', 'i'])
    >>> la[(0, 1)]
    array([3, 3, 3, 3])
    labels(['f', 'g', 'h', 'i'])
    >>> la[0, 1]
    array([3, 3, 3, 3])
    labels(['f', 'g', 'h', 'i'])
    >>> la[(0, 1),].labels
    [['a', 'b'], ['c', 'd', 'e'], ['f', 'g', 'h', 'i']]
    >>> np.nanmean(la, axis=(-2, -1))
    array([1.75, 1.  ])
    labels(['a', 'b'])
    >>> arr = np.arange(24).reshape((2, 3, 4))
    >>> labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
    >>> ad = LabeledArray(arr, labels)
    >>> ad[None, 'a'].labels
    [['1'], ['c', 'd', 'e'], ['f', 'g', 'h', 'i']]
    >>> ad['b', 0, np.array([[1,2], [0,3]])]
    array([[13, 14],
           [12, 15]])
    labels(['g-h', 'f-i']
           ['f-g', 'h-i'])
    >>> ad[:, ('d','e'),][..., ('g', 'h'),].labels
    [['a', 'b'], ['d', 'e'], ['g', 'h']]
    >>> ad['a', 'd', ('g', 'i', 'f'),]
    array([5, 7, 4])
    labels(['g', 'i', 'f'])

    Notes
    -----
    Multiple sequence advanced indices objects are not supported. If you want
     to use multiple sequence indices, you should use them one at a time.

    References
    ----------
    [1] https://numpy.org/doc/stable/user/basics.subclassing.html
    [2] https://numpy.org/doc/stable/user/basics.indexing.html
    """

    labels: list = []

    def __new__(cls, input_array, labels: list[tuple[str, ...], ...] = (),
                delimiter: str = '-', **kwargs):
        obj = np.asarray(input_array, **kwargs).view(cls)
        labels = list(labels)
        for i in range(obj.ndim):
            if len(labels) < i + 1:
                labels.append(tuple(range(obj.shape[i])))
        obj.labels = list(map(lambda lab: Labels(lab, delimiter), labels))
        assert tuple(map(len, obj.labels)) == obj.shape, \
            f"labels must have the same length as the shape of the array, " \
            f"instead got {tuple(map(len, obj.labels))} and {obj.shape}"
        return obj

    def __array_finalize__(self, obj, *args, **kwargs):
        if obj is None:
            return
        self.labels = getattr(obj, 'labels', kwargs.pop('labels', ()))
        super(LabeledArray, self).__array_finalize__(obj, *args, **kwargs)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(LabeledArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.labels,)
        # Return a tuple that replaces the parent's __setstate__
        # tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.labels = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(LabeledArray, self).__setstate__(state[0:-1])

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        la_inputs = (i for i in inputs if isinstance(i, LabeledArray))
        labels = next(la_inputs).labels.copy()
        inputs = tuple(i.view(np.ndarray) if isinstance(i, LabeledArray)
                       else i for i in inputs)
        if out is not None:
            kwargs['out'] = tuple(o.view(np.ndarray) if
                                  isinstance(o, LabeledArray)
                                  else o for o in out)
        if method == 'reduce':
            axis = kwargs.get('axis', None)
            if axis is None:
                axis = range(inputs[0].ndim)
            elif np.isscalar(axis):
                axis = (axis,)
            else:
                axis = tuple(axis)
            i = 0
            for ax in axis:
                if ax > 0:
                    ax -= i
                labels = list(labels)
                if kwargs.get('keepdims', False):
                    labels[ax] = ("-".join(labels[ax]),)
                else:
                    labels.pop(ax)
                    i += 1
                labels = tuple(labels)

        outputs = super(LabeledArray, self).__array_ufunc__(
            ufunc, method, *inputs, **kwargs)
        if isinstance(outputs, tuple):
            outputs = tuple(LabeledArray(o, labels)
                            if isinstance(o, np.ndarray)
                            else o for o in outputs)
        elif isinstance(outputs, np.ndarray):
            outputs = LabeledArray(outputs, labels)
        return outputs

    @property
    def T(self):
        return LabeledArray(self.__array__().T, self.labels[::-1])

    def swapaxes(self, axis1, axis2):
        new = list(self.labels)
        new[axis1], new[axis2] = new[axis2], new[axis1]
        arr = super(LabeledArray, self).swapaxes(axis1, axis2)
        return LabeledArray(arr, new)

    def transpose(self, axes):
        axes = np._core.numeric.normalize_axis_tuple(axes, self.ndim)
        new_labels = [self.labels[i] for i in axes]
        arr_t = super(LabeledArray, self).transpose(axes)
        return LabeledArray(arr_t, new_labels)

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> 'LabeledArray':
        """Create a LabeledArray from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary to convert to a LabeledArray.

        Returns
        -------
        LabeledArray
            The LabeledArray created from the dictionary.

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1}}}
        >>> LabeledArray.from_dict(data, dtype=int) # doctest: +ELLIPSIS
        array([[[1]]])
        labels(['a']
               ['b']
               ['c'])
        >>> data = {'a': {'b': {'c': 1}}, 'd': {'b': {'c': 2, 'e': 3}}}
        >>> LabeledArray.from_dict(data) # doctest: +ELLIPSIS
        array([[[ 1., nan]],
        <BLANKLINE>
               [[ 2.,  3.]]])
        labels(['a', 'd']
               ['b']
               ['c', 'e'])
        >>> data = {'a': {'b': np.array([[1, 2, 3]]), 'c' : [[4, 5], [6, 7]]},}
        >>> LabeledArray.from_dict(data) # doctest: +ELLIPSIS
        array([[[[ 1.,  2.,  3.],
                 [nan, nan, nan]],
        <BLANKLINE>
                [[ 4.,  5., nan],
                 [ 6.,  7., nan]]]])
        labels(['a']
               ['b', 'c']
               ['0', '1']
               ['0', '1', '2'])
        >>> data = {'b': {'c': 1, 'd': 2, 'e': 3}, 'f': {'c': 4, 'e': 6}}
        >>> LabeledArray.from_dict(data)
        array([[ 1.,  2.,  3.],
               [ 4., nan,  6.]])
        labels(['b', 'f']
               ['c', 'd', 'e'])
        """

        keys = inner_all_keys(data)
        # each key layer is unique by definition
        # also non-homogenous shape sequence would have failed by now
        # example: {'c' : [[4, 5], [6]]}
        dtype = kwargs.pop('dtype', None)
        tmp = data
        if dtype is None:
            for key in keys:
                tmp = tmp[key[0]]
            dtype = get_float_type(type(tmp))

        shape = tuple(len(keys[i]) for i in range(len(keys)))
        #   try to create output array, fall back to memory map if too large
        try:
            arr = np.full(shape, np.nan, dtype=dtype)
        except MemoryError:
            arr = np.memmap('data.dat', dtype=dtype, mode='w+', shape=shape)
            arr[...] = np.nan

        # slightly faster than using keys[i].index(key), O(n+m) vs O(n*m)
        keys_dict = tuple({k: i for i, k in enumerate(ks)} for ks in keys)
        for k, v in iter_nest_dict(data):
            coords = tuple(keys_dict[i][key] for i, key in enumerate(k))
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v)
                coords += tuple(slice(0, s) for s in v.shape)
            arr[coords] = v
        return cls(arr, keys, **kwargs)

    @classmethod
    def from_signal(cls, sig: ieeg.Signal, **kwargs) -> 'LabeledArray':
        """Create a LabeledArray from a Signal.

        Parameters
        ----------
        sig : Signal
            The Signal to convert to a LabeledArray.

        Returns
        -------
        LabeledArray
            The LabeledArray created from the Signal.

        Examples
        --------
        >>> from bids import BIDSLayout
        >>> from ieeg.io import raw_from_layout
        >>> from ieeg.navigate import trial_ieeg
        >>> import sys
        >>> bids_root = mne.datasets.epilepsy_ecog.data_path()
        >>> layout = BIDSLayout(bids_root)
        >>> with mne.use_log_level(0):
        ...     raw = raw_from_layout(layout, subject="pt1", preload=True,
        ...     extension=".vhdr", verbose=False)
        >>> LabeledArray.from_signal(raw, dtype=float) # doctest: +ELLIPSIS
        array([[-8.98329883e-06,  8.20419238e-06,  7.42294287e-06, ...,
                 1.07177293e-09,  1.07177293e-09,  1.07177293e-09],
               [ 2.99222000e-04,  3.03518844e-04,  2.96878250e-04, ...,
                 3.64667153e-09,  3.64667153e-09,  3.64667153e-09],
               [ 2.44140953e-04,  2.30078469e-04,  2.19140969e-04, ...,
                 3.85053724e-10,  3.85053724e-10,  3.85053724e-10],
               ...,
               [ 1.81263844e-04,  1.74232594e-04,  1.56263875e-04, ...,
                 1.41283798e-08,  1.41283798e-08,  1.41283798e-08],
               [ 2.25390219e-04,  2.16015219e-04,  1.91405859e-04, ...,
                -2.91418821e-10, -2.91418821e-10, -2.91418821e-10],
               [ 3.14092313e-04,  3.71123375e-04,  3.91826437e-04, ...,
                 3.07457047e-08,  3.07457047e-08,  3.07457047e-08]])
        labels(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', ...
        >>> epochs = trial_ieeg(raw, "AD1-4, ATT1,2", (-1, 2), preload=True,
        ... verbose=False)
        >>> LabeledArray.from_signal(epochs, dtype=float) # doctest: +ELLIPSIS
        array([[[ 0.00021563,  0.00021563,  0.00020703, ..., -0.00051211,
                 -0.00051445, -0.00050351],
                [-0.00030586, -0.00030625, -0.00031171, ..., -0.00016054,
                 -0.00015976, -0.00015664],
                [-0.00010781, -0.00010469, -0.00010859, ...,  0.00026719,
                  0.00027695,  0.00030156],
                ...,
                [-0.00021483, -0.00021131, -0.00023084, ..., -0.00034295,
                 -0.00032381, -0.00031444],
                [-0.00052188, -0.00052852, -0.00053125, ..., -0.00046211,
                 -0.00047148, -0.00047891],
                [-0.00033708, -0.00028005, -0.00020934, ..., -0.00040934,
                 -0.00042341, -0.00040973]]])
        ...

        """

        arr = sig.get_data()
        match sig:
            case mne.io.base.BaseRaw():
                labels = [sig.ch_names, sig.times]
            case mne.BaseEpochs():
                events = events_in_order(sig)
                labels = [events, sig.ch_names, sig.times]
            case mne.evoked.Evoked():
                labels = [sig.ch_names, sig.times]
            case mne.time_frequency.EpochsTFR():
                events = events_in_order(sig)
                labels = [events, sig.ch_names, sig.freqs, sig.times]
            case mne.time_frequency.AverageTFR():
                labels = [sig.ch_names, sig.freqs, sig.times]
            case _:
                raise TypeError(f"Unexpected data type: {type(sig)}")
        return cls(arr, labels, **kwargs)

    def tofile(self, fid: str, **kwargs) -> None:
        """Save the LabeledArray to a file.

        Parameters
        ----------
        file : str
            The file to save the LabeledArray to.
        **kwargs
            Additional arguments to pass to np.save.

        Examples
        --------
        >>> arr = np.arange(24).reshape((2, 3, 4))
        >>> labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
        >>> la = LabeledArray(arr, labels)
        >>> la.tofile('data')
        >>> la2 = LabeledArray.fromfile('data')
        >>> la == la2
        True
        """
        files = {str(i): l for i, l in enumerate(self.labels)}
        np.save(fid + '.npy', self.__array__())
        np.savez(fid + '_labels.npz', **files)

    @classmethod
    def fromfile(cls, file: str, **kwargs) -> 'LabeledArray':
        """Create a LabeledArray from a file.

        Parameters
        ----------
        file : str
            The file to load the LabeledArray from.
        **kwargs
            Additional arguments to pass to np.load.

        Returns
        -------
        LabeledArray
            The LabeledArray created from the file.

        Examples
        --------
        >>> arr = np.arange(24).reshape((2, 3, 4))
        >>> labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
        >>> la = LabeledArray(arr, labels)
        >>> la.tofile('data')
        >>> la2 = LabeledArray.fromfile('data')
        >>> la == la2
        True
        """
        kwargs['allow_pickle'] = False
        files = np.load(file + '_labels.npz', **kwargs)
        labels = list(map(tuple, files.values()))
        return cls(np.load(file + '.npy', **kwargs), labels)

    def _parse_index(self, keys: list) -> list:
        ndim = self.ndim
        new_keys = [range(self.shape[i]) for i in range(ndim)]
        dim = 0
        newaxis_count = 0
        for i, key in enumerate(keys):
            key_type = type(key)
            if np.issubdtype(key_type, str):
                key = self.labels[dim - newaxis_count].find(key)
                keys[i] = key  # set original keys as well
            elif key is Ellipsis:
                num_ellipsis_dims = ndim - len(keys) + 1
                while dim < num_ellipsis_dims:
                    dim += 1
                continue
            elif key_type is slice:
                key = new_keys[dim][key]
            elif key is np.newaxis or key is None:
                new_keys.insert(dim, None)
                newaxis_count += 1
                dim += 1
                continue
            elif (key_type in (list, tuple) or
                  np.issubdtype(key_type, np.ndarray)):
                key = list(key)
                for j, k in enumerate(key):
                    if np.issubdtype(type(k), str):
                        key[j] = self.labels[dim - newaxis_count].find(k)
                if np.issubdtype(key_type, np.ndarray):
                    keys[i] = np.array(key)
                else:
                    keys[i] = key_type(key)
            elif np.isscalar(key):  # key should be an int
                while key < 0:
                    key += self.shape[dim - newaxis_count]
            else:
                raise TypeError(f"Unexpected key type: {key_type}")

            new_keys[dim] = key
            dim += 1
        return new_keys

    def _to_coords(self, orig_keys):

        if np.isscalar(orig_keys) or np.issubdtype(
                (dtype := getattr(orig_keys, 'dtype', None)), np.integer):
            keys = [orig_keys]
            l_keys = self._parse_index(keys)
            return keys[0], tuple(l_keys)
        elif dtype == np.bool_ and is_broadcastable(
                getattr(orig_keys, 'shape', ()), self.shape):
            l_keys = np.where(np.reshape(orig_keys, self.shape))
            return orig_keys, l_keys
        else:
            if isinstance(orig_keys, slice):
                keys = [orig_keys]
            else:
                keys = list(orig_keys)
            l_keys = self._parse_index(keys)
            return tuple(keys), tuple(l_keys)

    def __getitem__(self, orig_keys):
        keys, label_keys = self._to_coords(orig_keys)
        out = super(LabeledArray, self).__getitem__(keys)
        if out.ndim == 0:
            return out[()]

        # determine the new labels
        new_labels = [None] * out.ndim
        j = 0
        k = 0
        for i, label_key in enumerate(label_keys):

            if label_key is None:
                new_labels[i - k] = Labels(['1'])
                j += 1
            elif np.isscalar(label_key):  # basic indexing triggered
                k += 1
            elif i - k >= out.ndim:
                raise IndexError(f"Too many indices for array: "
                                 f"array is {out.ndim}-dimensional, "
                                 f"but {i + 1} were indexed")
            else:
                if isinstance(label_key, tuple):
                    label_key = np.asarray(label_key)
                labels = np.atleast_1d(np.squeeze(self.labels[i - j][label_key]
                                                  ))
                if labels.ndim > 1:
                    lab_list = labels.decompose()
                    new_labels[i - k:i - k + len(labels)] = lab_list
                    k += len(lab_list) - 1
                else:
                    new_labels[i - k] = labels

        if any(l_none := lab is None for lab in new_labels):
            raise IndexError(f"Too few indices for array: array is {out.ndim}"
                             f"-dimensional, but {sum(~l_none)} were indexed")

        setattr(out, 'labels', new_labels)
        return out

    def __setitem__(self, keys, value):
        keys, _ = self._to_coords(keys)
        super(LabeledArray, self).__setitem__(keys, value)

    def __repr__(self):
        return repr(self.__array__()) + f"\nlabels({self._label_formatter()})"

    def __str__(self):
        return str(self.__array__()) + f"\nlabels({self._label_formatter()})"

    def _label_formatter(self):
        def _liststr(x):
            return f"\n       ".join(x)

        return _liststr([str(lab) for lab in self.labels])

    def memory(self):
        size = self.nbytes
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if size < 1024.0 or unit == 'PiB':
                break
            size /= 1024.0
        return size, unit

    def __eq__(self, other):
        if isinstance(other, LabeledArray):
            return np.array_equal(self, other, True) and \
                all(np.array_equal(l1, l2) for l1, l2 in zip(self.labels,
                                                             other.labels))
        else:
            return self.__array__().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        out = {}
        for k, v in self.items():
            if len(self.labels) > 1:
                out[k] = v.to_dict()
            elif np.isnan(v).all():
                continue
            else:
                out[k] = v
        return out

    def items(self):
        return zip(self.keys(), self.values())

    def keys(self):
        return (lab for lab in self.labels[0])

    def values(self):
        return (a for a in self)

    def _reshape(self, shape, order='C') -> 'LabeledArray':
        """Reshape the array.

        Parameters
        ----------
        shape : tuple[int, ...]
            The new shape of the array.
        order : str, optional
            The order to reshape the array in, by default 'C'

        Returns
        -------
        LabeledArray
            The reshaped LabeledArray.

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1}}}
        >>> ad = LabeledArray.from_dict(data, dtype=int)
        >>> ad.labels
        [['a'], ['b'], ['c']]
        >>> ad._reshape((1, 1, 1)) # doctest: +SKIP
        array([[[1]]])
        labels(['a']
               ['b']
               ['c'])
        >>> arr = np.arange(24).reshape((2, 3, 4))
        >>> labels = [('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i')]
        >>> ad = LabeledArray(arr, labels)
        >>> ad._reshape((6, 4))
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11],
               [12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
        labels(['a-c', 'a-d', 'a-e', 'b-c', 'b-d', 'b-e']
               ['f', 'g', 'h', 'i'])
        >>> ad._reshape((6, 4), 'F').labels
        [['a-c', 'b-c', 'a-d', 'b-d', 'a-e', 'b-e'], ['f', 'g', 'h', 'i']]
        >>> ad._reshape((2, 12)).labels # doctest: +ELLIPSIS
        [['a', 'b'], ['c-f', 'c-g', 'c-h', 'c-i', 'd-f', 'd-g', 'd-h', 'd-i'...
        >>> arr = np.arange(10)
        >>> labels = [list(map(str, arr))]
        >>> ad = LabeledArray(arr, labels)
        >>> ad._reshape((2, 5)).labels
        [['0-1-2-3-4', '5-6-7-8-9'], ['0-5', '1-6', '2-7', '3-8', '4-9']]
        >>> ad._reshape((1, 2, 5)).labels # doctest: +ELLIPSIS
        [['0-1-2-3-4-5-6-7-8-9'], ['0-1-2-3-4', '5-6-7-8-9'], ['0-5', '1-6',...
        """
        new_array = super(LabeledArray, self).reshape(*shape, order=order)
        lab_mat = functools.reduce(lambda x, y: x @ y, self.labels)
        new_labels = lab_mat.reshape(*shape, order=order).decompose()
        return LabeledArray(new_array, new_labels)

    def combine(self, levels: tuple[int, int]) -> 'LabeledArray':
        """Combine any levels of a LabeledArray into the lower level

        Takes the input LabeledArray and rearranges its dimensions.

        Parameters
        ----------
        levels : tuple[int, int]
            The levels to combine, e.g. (0, 1) will combine the 1st and 2nd
            level of the array labels into one level at the 2nd level.
        delim : str, optional
            The delimiter to use when combining labels, by default '-'

        Returns
        -------
        LabeledArray
            The combined LabeledArray

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1}}}
        >>> ad = LabeledArray.from_dict(data, dtype=int)
        >>> ad.combine((0, 2))
        array([[1]])
        labels(['b']
               ['a-c'])
        >>> ad2 = LabeledArray([[[1,2],[3,4]],[[5,6],[7,8]]],
        ... labels=[('a', 'b'), ('c', 'd'), ('e', 'f')])
        >>> ad2['a', : , 'e']
        array([1, 3])
        labels(['c', 'd'])
        >>> ad2.combine((0, 2))
        array([[1, 2, 5, 6],
               [3, 4, 7, 8]])
        labels(['c', 'd']
               ['a-e', 'a-f', 'b-e', 'b-f'])
        >>> np.mean(ad2.combine((0, 2)), axis=1)
        array([3.5, 5.5])
        labels(['c', 'd'])
        >>> np.mean(ad2, axis=(0, 2))
        array([3.5, 5.5])
        labels(['c', 'd'])
        """

        assert levels[0] >= 0, "first level must be >= 0"
        assert levels[1] > levels[0], "second level must be > first level"

        new_labels = list(self.labels).copy()
        new_labels.pop(levels[0])

        new_labels[levels[1] - 1] = (
                self.labels[levels[0]] @ self.labels[levels[1]]).flatten()

        all_idx = ([slice(None) if i != levels[0] else sl for i in
                    range(self.ndim)] for sl in range(self.shape[levels[0]]))

        arrs = [self.__array__()[tuple(idx)] for idx in all_idx]
        new_array = concatenate_arrays(arrs, axis=levels[1] - 1)

        return LabeledArray(new_array, new_labels, dtype=self.dtype)

    def take(self, indices, axis=None, **kwargs):
        """Take elements from an array along an axis.

        This function does not support the out argument.

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract.
        axis : int, optional
            The axis over which to select values, by default None.
        kwargs : dict
            Additional keyword arguments to pass to np.take.

        Returns
        -------
        LabeledArray
            The LabeledArray with the selected elements.

        Examples
        --------
        >>> arr = np.arange(24).reshape((2, 3, 4))
        >>> labels = [('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i')]
        >>> ad = LabeledArray(arr, labels)
        >>> ad.take([0, 2], axis=1).labels
        [['a', 'b'], ['c', 'e'], ['f', 'g', 'h', 'i']]
        >>> np.take_along_axis(ad, np.array([[[0, 1]]]), axis=2).labels
        [['a', 'b'], ['c', 'd', 'e'], ['f', 'g']]
        >>> np.take(ad, np.array([[0,2], [1,3]]), axis=2).labels
        [['a', 'b'], ['c', 'd', 'e'], ['f-h', 'g-i'], ['f-g', 'h-i']]
        >>> np.take(ad, np.array(['f','g']), axis=2)
        array([[[ 0,  1],
                [ 4,  5],
                [ 8,  9]],
        <BLANKLINE>
               [[12, 13],
                [16, 17],
                [20, 21]]])
        labels(['a', 'b']
               ['c', 'd', 'e']
               ['f', 'g'])
        >>> np.take(ad, 'f', axis=2).labels
        [['a', 'b'], ['c', 'd', 'e']]
        >>> np.take(ad, ('c','e'), axis=1).labels
        [['a', 'b'], ['c', 'e'], ['f', 'g', 'h', 'i']]
        """

        idx = [slice(None)] * self.ndim
        if isinstance(indices, str):
            indices = self.labels[axis].find(indices)
        elif not isinstance(indices, int):
            indices = np.array(indices)

        if axis is None:
            return self.flat[indices]
        elif isinstance(axis, int):
            if not isinstance(indices, int):
                if indices.dtype.kind == 'U':
                    indices = np.array(
                        [self.labels[axis].find(idx) for idx in indices])
            idx[axis] = indices
        elif len(indices) == len(axis):
            for i, ax in enumerate(axis):
                if indices.dtype.kind == 'U':
                    indices = np.array(
                        [self.labels[ax].find(idx) for idx in indices])
                idx[ax] = indices[i]
        else:
            raise ValueError("indices and axis must have the same length")

        out = super(LabeledArray, self).take(indices, axis, **kwargs)
        labels = [l[i] for i, l in zip(idx, self.labels)
                  if not np.isscalar(l[i])]
        for i, l in enumerate(labels):
            if l.ndim > 1:
                labels = labels[:i] + l.decompose()
        return LabeledArray(out, labels, dtype=self.dtype)

    def dropna(self) -> 'LabeledArray':
        """Remove all nan values from the array.

        Scans each column along any axis and removes all rows that contain
        only nan values.

        Returns
        -------
        LabeledArray
            The array with all nan values removed.

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1., 'd': np.nan}}}
        >>> ad = LabeledArray.from_dict(data)
        >>> ad.dropna()
        array([[[1.]]])
        labels(['a']
               ['b']
               ['c'])
        >>> ad2 = LabeledArray([[[1,2],[3,4]],[[4,5],[6,7]],
        ... [[np.nan, np.nan], [np.nan, np.nan]]])
        >>> ad2.dropna()
        array([[[1., 2.],
                [3., 4.]],
        <BLANKLINE>
               [[4., 5.],
                [6., 7.]]])
        labels(['0', '1']
               ['0', '1']
               ['0', '1'])
        """
        new_labels = list(self.labels)
        idx = []
        for i in range(self.ndim):
            axes = tuple(j for j in range(self.ndim) if j != i)
            mask = np.all(np.isnan(np.array(self)), axis=axes)
            if np.any(mask):
                new_labels[i] = tuple(np.array(new_labels[i])[~mask])
            idx.append(~mask)
        index = np.ix_(*idx)
        return self[index]

    def concatenate(self, other: 'LabeledArray', axis: int = 0,
                    mismatch: str = 'raise', ids: tuple[str, str] = ('0', '1'),
                    **kwargs) -> 'LabeledArray':
        """Concatenate two LabeledArrays along an axis.

        Parameters
        ----------
        other : LabeledArray
            The LabeledArray to concatenate with.
        axis : int, optional
            The axis to concatenate along, by default 0.
        mismatch : str, optional
            What to do if the number of labels are not the same, 'raise'
            (default) will raise a ValueError, 'shrink' will shrink the labels
            to the smallest size, and 'expand' (not implemented) will expand
            the labels to the largest size, filling in with NaNs.
        ids : tuple[str, str], optional
            The identifiers for the two arrays, used to create unique labels
        kwargs : dict
            Additional keyword arguments to pass to np.concatenate.

        Returns
        -------
        LabeledArray
            The concatenated LabeledArray.

        Examples
        --------
        >>> arr1 = LabeledArray([[1, 2],[3, 4]],
        ... labels=[('a', 'b'), ('c', 'd')])
        >>> arr2 = LabeledArray([[5, 6],[7, 8]],
        ... labels=[('a', 'b'), ('c', 'd')])
        >>> arr1.concatenate(arr2, axis=0)
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
        labels(['a-0', 'b-0', 'a-1', 'b-1']
               ['c', 'd'])
        >>> arr3 = LabeledArray([[5, 6, 9],[7, 8, 10]],
        ... labels=[('a', 'b'), ('c', 'd', 'e')])
        >>> arr4 = LabeledArray([[1, 2, 3],[3, 4, 5]],
        ... labels=[('a', 'b'), ('c', 'e', 'd')])
        >>> arr3.concatenate(arr4, axis=0)
        array([[ 5,  6,  9],
               [ 7,  8, 10],
               [ 1,  3,  2],
               [ 3,  5,  4]])
        labels(['a-0', 'b-0', 'a-1', 'b-1']
               ['c', 'd', 'e'])
        >>> arr2.concatenate(arr4, axis=0) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: When mismatch is 'raise', the base array must the same s...
        >>> arr2.concatenate(arr4, 0, mismatch='shrink')
        array([[5, 6],
               [7, 8],
               [1, 3],
               [3, 5]])
        labels(['a-0', 'b-0', 'a-1', 'b-1']
               ['c', 'd'])
        >>> arr3.concatenate(arr1, 0, mismatch='shrink') # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        NotImplementedError: Base array must the same size or smaller than i...
        Base size:(2, 3), Input size: (2, 2)
        >>> arr1.concatenate(arr3, 0, mismatch='expand')
        array([[ 1.,  2., nan],
               [ 3.,  4., nan],
               [ 5.,  6.,  9.],
               [ 7.,  8., 10.]])
        labels(['0-a', '0-b', '1-a', '1-b']
               ['c', 'd', 'e'])
        """

        while axis < 0:
            axis += self.ndim

        if mismatch == 'expand':
            ids = tuple(map(str, ids))
            all_dict = {ids[0]: self.to_dict(), ids[1]: other.to_dict()}
            combined = combine(all_dict, (0, axis + 1),
                               self.labels[0].delimiter)
            return LabeledArray.from_dict(combined)

        new_labels = list(self.labels)
        idx = [slice(None)] * self.ndim
        new = np.hstack((self.labels[axis], other.labels[axis]))
        for i in range(self.ndim):
            if i == axis:
                if not is_unique(new):
                    new_labels[i] = make_array_unique(
                        new.astype(str), self.labels[i].delimiter)
                else:
                    new_labels[i] = new
            elif not (is_unique(new_labels[i]) and is_unique(other.labels[i])):
                raise NotImplementedError(
                    "Cannot concatenate arrays with non-unique labels "
                    f"{new_labels[i]}, {other.labels[i]}")
            elif self.shape[i] == other.shape[i]:
                if np.any(self.labels[i] != other.labels[i]):
                    idx[i] = get_subset_reorder_indices(
                        other.labels[i], self.labels[i])
            elif mismatch == 'raise':
                raise ValueError(
                    "When mismatch is 'raise', the base array must the same "
                    "size as the input array in all but the concatination "
                    f"axis, but along dimension {i} the base array has size "
                    f"{self.shape[i]} and the input array has size "
                    f"{other.shape[i]}")
            elif self.labels[i].shape[0] < other.labels[i].shape[0]:
                if mismatch == 'shrink':
                    idx[i] = get_subset_reorder_indices(
                        other.labels[i], self.labels[i])
                else:
                    raise NotImplementedError(
                        f"No method associated with mismatch = '{mismatch}',"
                        " try setting mismatch to 'shrink' or 'raise'")
            elif self.labels[i].shape[0] > other.labels[i].shape[0]:
                raise NotImplementedError(
                    "Base array must the same size or smaller than input "
                    "array in all but the concatination axes. \nBase size:"
                    f"{self.shape}, Input size: {other.shape}")
            else:
                raise ValueError("Unexpected error")

        reordered = other.__array__()[tuple(idx)]
        out = np.concatenate((self.__array__(), reordered), axis, **kwargs)
        return LabeledArray(out, new_labels, dtype=self.dtype)

    # def swapaxes(self):


def is_unique(arr: np.ndarray) -> bool:
    """Check if an array is unique.

    Parameters
    ----------
    arr : np.ndarray
        The array to check.

    Returns
    -------
    bool
        Whether the array is unique.

    Examples
    --------
    >>> is_unique(np.array([1, 2, 3]))
    True
    >>> is_unique(np.array([1, 2, 2]))
    False
    """
    return np.unique(arr).shape[0] == np.prod(arr.shape)


class Labels(np.char.chararray):
    """A class for storing labels for a LabeledArray.

    Examples
    --------
    >>> Labels(['D21']) @ Labels(['a', 'b', 'c',])
    [['D21-a', 'D21-b', 'D21-c']]
    """
    delimiter: str

    # __slots__ = ['delimiter', '__dict__']

    def __new__(cls, input_array: ArrayLike, delim: str = '-'):
        obj = np.asarray(input_array, dtype=str).view(cls)
        setattr(obj, 'delimiter', delim)
        return obj

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Labels, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.delimiter,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our
        # own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.delimiter = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Labels, self).__setstate__(state[0:-1])

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.delimiter = getattr(obj, 'delimiter', '-')

    def __str__(self):
        return self.tolist().__str__()

    def __repr__(self):
        return self.tolist().__repr__()

    def __matmul__(self, other):
        if not isinstance(other, Labels):
            raise NotImplementedError("Only Labels @ Labels is supported")
        s_str, o_str = self.astype(str), other.astype(str)

        # Convert the arrays to 2D
        s_str_2d = s_str[..., None]
        o_str_2d = o_str[None]

        # Use broadcasting to create a result array with combined strings
        result = s_str_2d + o_str_2d
        return result

    def __add__(self, other):
        result = self.view(np.char.chararray).__add__(
            self.delimiter).__add__(other.view(np.char.chararray))
        return Labels(result)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.matmul:
            # Call the __matmul__ method
            return self.__matmul__(*inputs)
        elif ufunc is np.add:
            # Call the __add__ method
            return self.__add__(*inputs)
        # Convert all inputs to base class (np.char.chararray) for computation
        inputs = [i.view(np.char.chararray) if isinstance(i, Labels)
                  else i for i in inputs]
        # Perform the ufunc operation
        out = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        # Return the result as a Labels object
        return Labels(out)

    def split(
        self,
        sep: str = None,
        maxsplit: int = -1,
    ):
        """
        Return a list of the words in the string, using sep as the delimiter
         string.

        sep
            The delimiter according which to split the string.
            None (the default value) means split according to the given
            delimiter
        maxsplit
            Maximum number of splits to do.
            -1 (the default value) means no limit.

        Examples
        --------
        >>> Labels(['a-b-c', 'd-e-f']).split('-')
        array([['a', 'b', 'c'],
               ['d', 'e', 'f']], dtype='<U1')
        >>> Labels(['a-b-c', 'd-e-f'], '-').split()
        array([['a', 'b', 'c'],
               ['d', 'e', 'f']], dtype='<U1')
        """
        if sep is None:
            sep = self.delimiter
        return np.array(super(Labels, self).split(sep, maxsplit).tolist())

    def decompose(self) -> list['Labels', ...]:
        """Decompose a Labels object into a list of 1d Labels objects.

        Examples
        --------
        >>> Labels(['a-d', 'a-c', 'b-d', 'b-c']).reshape(2,2).decompose()
        [['a', 'b'], ['d', 'c']]
        >>> Labels(['a-c-e', 'a-c-f', 'a-d-e', 'a-d-f', 'b-c-e', 'b-c-f',
        ... 'b-d-e', 'b-d-f']).reshape(2,2,2).decompose()
        [['a', 'b'], ['c', 'd'], ['e', 'f']]
        >>> (Labels(['a','b','c']) @ Labels(['d','e','f','g'])).reshape(
        ... 2,6).decompose() # doctest: +ELLIPSIS
        [['a-d-a-e-a-f-a-g-b-d-b-e', 'b-f-b-g-c-d-c-e-c-f-c-g'], ['a-d-b-f'...
        """
        new_labels = [[None for _ in range(s)] for s in self.shape]
        for i, dim in enumerate(self.shape):
            for j in range(dim):
                row = np.take(self, j, axis=i).flatten().astype(str)
                splitted = row.split(self.delimiter)
                common = functools.reduce(np.intersect1d, splitted)
                if len(common) == 0:
                    common = np.unique(row).tolist()
                new_labels[i][j] = self.delimiter.join(common)
            new_labels[i] = make_array_unique(np.array(new_labels[i]),
                                              self.delimiter)
        return list(map(Labels, new_labels))

    def find(self, value) -> int | tuple[int]:
        """Get the index of the first instance of a value in the Labels"""
        idx = np.where(self == value)[0]
        if (n := len(idx)) == 0:
            if self.delimiter in self[0]:
                splitlist = np.char.split(self, self.delimiter)
                for i in range(len(splitlist[0])):
                    try:
                        return Labels([s[i] for s in splitlist]).find(value)
                    except IndexError:
                        continue
            raise IndexError(f"{value} not found in {self}")
        elif n == 1:
            return int(idx[0])
        else:
            return tuple(map(int, idx))

    def join(self, axis: int = None):
        """Join the labels into a single string using the delimiter

        Parameters
        ----------
        axis : int, optional
            The axis to join along, by default None

        Examples
        --------
        >>> Labels(['a', 'b', 'c']).join()
        'a-b-c'
        >>> Labels(['a', 'b', 'c']).reshape(1,3).join()
        'a-b-c'
        >>> Labels([['a','b'],['c','d']]).join()
        'a-b-c-d'
        >>> Labels([['a','b'],['c','d']]).join(axis=0)
        ['a-b', 'c-d']
        >>> Labels([['a','b'],['c','d']], '').join(axis=1)
        ['ac', 'bd']
        """
        if axis is None:
            return self.delimiter.join(self.flat)
        else:
            labs = self.swapaxes(0, axis)
            return Labels([lab.join() for lab in labs], self.delimiter)


def make_array_unique(arr: np.ndarray, delimiter: str) -> np.ndarray:
    """Make an array unique by appending a number to duplicate values.

    Parameters
    ----------
    arr : np.ndarray
        The array to make unique.
    delimiter : str
        The delimiter to use when appending a number to duplicate values.

    Returns
    -------
    np.ndarray
        The unique array.

    Examples
    --------
    >>> arr = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
    >>> make_array_unique(arr, '-')
    array(['a-0', 'b-0', 'c-0', 'a-1', 'b-1', 'c-1'], dtype='<U3')
    >>> make_array_unique(arr[:-1], '-')
    array(['a-0', 'b-0', 'c', 'a-1', 'b-1'], dtype='<U3')
    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    >>> make_array_unique(arr, '-')
    array(['a', 'b', 'c', 'd', 'e', 'f'], dtype='<U1')
    """
    unique, inverse = np.unique(arr, return_inverse=True)
    if len(unique) == len(arr):
        return arr
    counts = np.bincount(inverse)
    max_dtype = np.max([len(u) for u in unique]) + 1 + len(str(max(counts)))
    out = np.empty_like(arr, dtype=f'<U{max_dtype}')
    for i, (u, c) in enumerate(zip(unique, counts)):
        if c == 1:
            out[inverse == i] = u
        else:
            indices = np.where(arr == u)[0]
            for j, index in enumerate(indices):
                out[index] = f"{u}{delimiter}{j}"
    return out


def is_broadcastable(shp1: tuple[int, ...], shp2: tuple[int, ...]):
    """Check if two shapes are broadcastable.

    Parameters
    ----------
    shp1 : tuple[int, ...]
        The first shape.
    shp2 : tuple[int, ...]
        The second shape.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_broadcastable((2, 3), (2, 3))
    True
    >>> is_broadcastable((2, 3), (3, 2))
    False
    >>> is_broadcastable((2, 3), (2, 1))
    True
    """

    ndim1 = len(shp1)
    ndim2 = len(shp2)
    if ndim1 < ndim2:
        shp1 += (1,) * (ndim2 - ndim1)
    elif ndim2 < ndim1:
        shp2 += (1,) * (ndim1 - ndim2)

    for a, b in zip(shp1, shp2):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def get_subset_reorder_indices(array1, array2):
    """Get indices to reorder array1 to match array2"""
    o = [np.where(array1 == i)[0][0] for i in array2 if i in array1]
    return np.array(o)


def add_to_list_if_not_present(lst: list, element: Iterable):
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


def inner_all_keys(data: dict, keys: list = None, lvl: int = 0):
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
            inner_all_keys(d, keys, lvl + 1)
    elif isinstance(data, (np.ndarray, list, tuple)):
        data = np.atleast_1d(data)
        rows = range(data.shape[0])
        if len(keys) < lvl + 1:
            keys.append(list(rows))
        else:
            add_to_list_if_not_present(keys[lvl], rows)
        if len(data.shape) > 1:
            if not np.isscalar(data[0]):
                inner_all_keys(data[0], keys, lvl + 1)
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")
    return tuple(map(tuple, keys))


def get_float_type(int_type):
    if int_type == np.int16:
        return np.float16
    elif int_type == np.int32:
        return np.float32
    elif int_type == np.int64 or int_type is int:
        return np.float64
    elif np.issubdtype(int_type, np.floating):
        return int_type
    else:
        raise ValueError("Unsupported integer type:" + str(int_type))


def _combine_arrays(*arrays, delim: str = '-') -> np.ndarray:
    # Create a meshgrid of indices
    grids = np.meshgrid(*arrays, indexing='ij')

    # Combine the grids into a single array with string concatenation
    result = np.core.defchararray.add(grids[0], delim)
    for grid in grids[1:]:
        result = np.core.defchararray.add(result, grid)

    return result


def combine(data: dict, levels: tuple[int, int], delim: str = '-') -> dict:
    """Combine any levels of a nested dict into the lower level

    Takes the input nested dict and rearranges the top and bottom
    sub-dictionary.

    Parameters
    ----------
    data: dict
        The nested dict to combine
    levels: tuple[int, int]
        The levels to combine, e.g. (0, 1) will combine the 1st and 2nd level
        of the dict keys into one level at the 2nd level.
    delim: str, optional
        The delimiter to use when combining keys, by default '-'

    Returns
    -------
    dict
        The combined dict

    Examples
    --------
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


def stack_la(arrays: tuple[LabeledArray, ...], new_labels: list[str, ...]
             ) -> LabeledArray:
    """Stack a sequence of LabeledArrays along a new axis.

    Parameters
    ----------
    arrays : LabeledArray
        The LabeledArrays to stack.
    new_labels : Labels
        The new labels for the stacked axis.

    Returns
    -------
    LabeledArray
        The stacked LabeledArray.

    Examples
    --------
    >>> arr1 = LabeledArray([[1, 2],[3, 4]], labels=[('a', 'b'), ('c', 'd')])
    >>> arr2 = LabeledArray([[5, 6, 7],[7, 8, 9]],
    ... labels=[('a', 'b'), ('c', 'd', 'e')])
    >>> stack_la((arr1, arr2), ['1', '2'])
    array([[[ 1.,  2., nan],
            [ 3.,  4., nan]],
    <BLANKLINE>
           [[ 5.,  6.,  7.],
            [ 7.,  8.,  9.]]])
    labels(['1', '2']
           ['a', 'b']
           ['c', 'd', 'e'])
    """
    new_array = concatenate_arrays([a.__array__() for a in arrays], None)

    # get the longest labels in each axis
    new_labels = [Labels(new_labels)]
    for i in range(new_array.ndim - 1):
        new_labels.append(max((a.labels[i] for a in arrays), key=len))

    return LabeledArray(new_array, new_labels)


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

    Examples
    --------
    >>> data = np.array([0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
    >>> get_elbow(data)
    4
    >>> data = np.array([1, 2, 3, 4, 5, 4.5, 4, 3.5, 3, 2, 1])
    >>> get_elbow(data)
    4
    """
    nPoints = len(data)
    allCoord = np.vstack((range(nPoints), data)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm,
                                                  (nPoints, 1)), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    # set distance to points below lineVec to 0
    distToLine[vecToLine[:, 1] < 0] = 0
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def events_in_order(inst: mne.BaseEpochs) -> list[str]:
    ids = {v: k for k, v in inst.event_id.items()}
    return [ids[e[2]] for e in inst.events]


if __name__ == "__main__":
    # import os
    # from ieeg.io import get_data
    # import mne
    #
    # conds = {"resp": ((-1, 1), "Response/LS"), "aud_ls": ((-0.5, 1.5),
    #                                                       "Audio/LS"),
    #          "aud_lm": ((-0.5, 1.5), "Audio/LM"), "aud_jl": ((-0.5, 1.5),
    #                                                          "Audio/JL"),
    #          "go_ls": ((-0.5, 1.5), "Go/LS"), "go_lm": ((-0.5, 1.5), "Go/LM")
    #          "go_jl": ((-0.5, 1.5), "Go/JL")}
    # task = "SentenceRep"
    # root = os.path.expanduser("~/Box/CoganLab")
    # # layout = get_data(task, root=root)
    # folder = 'stats_old'
    # mne.set_log_level("ERROR")
    #
    # arr = np.arange(24).reshape((2, 3, 4))
    # labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
    # ad = LabeledArray(arr, labels)
    # Labels(['a', 'b', 'c']) @ Labels(['d', 'e', 'f'])
    #
    # labels = Labels(np.arange(1000))
    # l2d = labels @ labels
    # x = l2d.reshape((10, -1)).decompose()
    # x = np.moveaxis(ad, 0, 1)

    test_list = ["delay/word/5", "delay/word/6", "delay/word/7",
                 "stim/word/5", "stim/word/6", "stim/word/7",]
    labels = Labels(test_list, delim="/")
    functools.reduce(np.setdiff1d, labels.split())


def _cat_test():
    """Test concatenation of arrays

    Concatenate a list of arrays along a given axis.

    Parameters
    ----------
    arrays
    axis

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(legacy='1.21')
    >>> a = np.array([[1, 2, 3]])
    >>> b = np.array([[4, 5]])
    >>> c = np.array([[6, 7, 8, 9]])
    >>> concatenate_arrays([a, b, c])
    array([[ 1.,  2.,  3., nan],
           [ 4.,  5., nan, nan],
           [ 6.,  7.,  8.,  9.]])
    >>> concatenate_arrays([a, b, c], axis=1)
    array([[1., 2., 3., 4., 5., 6., 7., 8., 9.]])
    """
