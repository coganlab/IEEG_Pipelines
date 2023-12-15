from collections.abc import Iterable
import functools

import mne

from ieeg.calc.reshape import concatenate_arrays
from ieeg import Signal

import numpy as np
from numpy.matlib import repmat
from numpy.typing import ArrayLike
from numba import njit, extending, types, vectorize


def iter_nest_dict(d: dict, _lvl: int = 0, _coords=()):
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
    >>> from ieeg.calc.mat import LabeledArray
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

    labels: list

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
        return LabeledArray(super(LabeledArray, self).swapaxes(axis1, axis2),
                            new)

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
        """

        arr = inner_array(data)
        keys = inner_all_keys(data)
        return cls(arr, keys, **kwargs)

    @classmethod
    def from_signal(cls, sig: Signal, **kwargs) -> 'LabeledArray':
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
        labels(['AD1-4, ATT1,2']
               ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', ...
               [-1.0, -0.999, -0.998, -0.997, -0.996, -0.995, -0.994, ...

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
        if np.isscalar(orig_keys):
            keys = [orig_keys]
            l_keys = self._parse_index(keys)
            return keys[0], tuple(l_keys)
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

    def prepend_labels(self, pre: str, level: int) -> 'LabeledArray':
        """Prepend a string to all labels at a given level.

        Parameters
        ----------
        pre : str
            The string to prepend to all labels.
        level : int
            The level to prepend the string to.

        Returns
        -------
        LabeledArray
            The LabeledArray with the prepended labels.

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1}}}
        >>> ad = LabeledArray.from_dict(data, dtype=int)
        >>> ad.prepend_labels('pre-', 1) # doctest: +ELLIPSIS
        array([[[1]]])
        labels(['a']
               ['pre-b']
               ['c'])
        """
        assert 0 <= level < self.ndim, "level must be >= 0 and < ndim"
        self.labels[level] = tuple(pre + lab for lab in self.labels[level])
        return LabeledArray(self.view(np.ndarray), self.labels)

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

        arrs = [self.__array__()[*idx] for idx in all_idx]
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
        """

        idx = [slice(None)] * self.ndim

        if axis is None:
            return self.flat[indices]
        elif isinstance(axis, int):
            idx[axis] = indices
        elif len(indices) == len(axis):
            for i, ax in enumerate(axis):
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
        labels([0, 1]
               [0, 1]
               [0, 1])
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
                    **kwargs) -> 'LabeledArray':
        """Concatenate two LabeledArrays along an axis.

        Parameters
        ----------
        other : LabeledArray
            The LabeledArray to concatenate with.
        axis : int, optional
            The axis to concatenate along, by default 0.
        kwargs : dict
            Additional keyword arguments to pass to np.concatenate.

        Returns
        -------
        LabeledArray
            The concatenated LabeledArray.

        Examples
        --------
        >>> arr1 = LabeledArray([[1,2],[3,4]], labels=[('a', 'b'), ('c', 'd')])
        >>> arr2 = LabeledArray([[5,6],[7,8]], labels=[('a', 'b'), ('c', 'd')])
        >>> arr1.concatenate(arr2, axis=0)
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
        labels(['a-0', 'b-0', 'a-1', 'b-1']
               ['c', 'd'])
        """
        new_labels = list(self.labels)
        new = np.hstack((self.labels[axis], other.labels[axis]))
        if len(set(new)) != len(new):
            new_labels[axis] = _make_array_unique(
                new.astype(str), self.labels[axis].delimiter)
        else:
            new_labels[axis] = new
        return LabeledArray(np.concatenate(
            (self.__array__(), other.__array__()), axis, **kwargs),
            new_labels, dtype=self.dtype)


class Labels(np.ndarray):
    """A class for storing labels for a LabeledArray."""
    delimiter: str
    # __slots__ = ['delimiter', '__dict__']

    def __new__(cls, input_array: ArrayLike, delim: str = '-'):
        obj = np.asarray(input_array).view(cls)
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
        o_str_2d = np.char.add(self.delimiter, o_str[None])

        # Use broadcasting to create a result array with combined strings
        result = np.char.add(s_str_2d, o_str_2d)
        return Labels(result)

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
                common = _longest_common_substring(tuple(map(
                    lambda x: tuple(x.split(self.delimiter, )), row)))
                if len(common) == 0:
                    common = np.unique(row).tolist()
                new_labels[i][j] = self.delimiter.join(common)
            new_labels[i] = _make_array_unique(np.array(new_labels[i]),
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


def _make_array_unique(arr: np.ndarray, delimiter: str) -> np.ndarray:
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
    >>> _make_array_unique(arr, '-')
    array(['a-0', 'b-0', 'c-0', 'a-1', 'b-1', 'c-1'], dtype='<U3')
    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    >>> _make_array_unique(arr, '-')
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


def _longest_common_substring(strings: tuple[tuple[str]]) -> tuple[str]:
    matrix = [[] for _ in range(len(strings))]
    for i in range(len(strings) - 1):
        matrix[i] = _lcs(strings[i], strings[i + 1])
    else:
        matrix[-1] = [True for _ in range(len(strings[-1]))]
    return np.array(strings[0])[np.all(matrix, axis=0)].tolist()


@functools.lru_cache(None)
def _lcs(s1: tuple, s2: tuple) -> list[bool]:
    matrix = [False for _ in range(len(s1))]
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            matrix[i] = True
    return matrix


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


@extending.overload(add_to_list_if_not_present)
def add_jit(lst: list, element: list):
    seen = set(lst)
    for x in element:
        if not (x in seen or seen.add(x)):
            lst.append(x)


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


def combine_arrays(*arrays, delim: str = '-') -> np.ndarray:
    # Create a meshgrid of indices
    grids = np.meshgrid(*arrays, indexing='ij')

    # Combine the grids into a single array with string concatenation
    result = np.core.defchararray.add(grids[0], delim)
    for grid in grids[1:]:
        result = np.core.defchararray.add(result, grid)

    return result


def inner_array(data: dict | np.ndarray) -> np.ndarray | None:
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
    array([[[1.]]])
    >>> data = {'a': {'b': {'c': 1}}, 'd': {'b': {'c': 2, 'e': 3}}}
    >>> inner_array(data)
    array([[[ 1., nan]],
    <BLANKLINE>
           [[ 2.,  3.]]])
    """
    if np.isscalar(data):
        return data
    elif isinstance(data, dict):
        gen_arr = (inner_array(d) for d in data.values())
        arr = [a for a in gen_arr if a is not None]
        if len(arr) > 0:
            return concatenate_arrays(arr, axis=None)
    # elif not isinstance(data, np.ndarray):
    #     raise TypeError(f"Unexpected data type: {type(data)}")

    # Call np.atleast_1d once and store the result in a variable
    data_1d = np.atleast_1d(data)

    # Use the stored result to check the length of data
    if len(data_1d) == 0:
        return
    elif len(data_1d) == 1:
        return data
    else:
        return np.array(data)


@njit(nogil=True, cache=True)
def inner_dict(data: np.ndarray | dict) -> dict:
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
    >>> dict(inner_dict(data)) # doctest: +ELLIPSIS +SKIP
    {0: DictType[int64,DictType[int64,int32]<iv=None>]<iv=None>({0: {0: 1}})}
    >>> data = np.array([[[1, np.nan]],
    ...                  [[2, 3]]])
    >>> dict(inner_dict(data)) # doctest: +ELLIPSIS +SKIP
    {0: DictType[int64,DictType[int64,float64]<iv=None>]<iv=None>({0: {0: ...
    """
    if isinstance(data, dict):
        return data
    elif hasattr(data, 'ndim'):
        result = {}
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
    np.array([range(nPoints), data])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * repmat(
        lineVecNorm, nPoints, 1), axis=1)
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
    import os
    from ieeg.io import get_data
    import mne
    conds = {"resp": ((-1, 1), "Response/LS"), "aud_ls": ((-0.5, 1.5),
                                                          "Audio/LS"),
             "aud_lm": ((-0.5, 1.5), "Audio/LM"), "aud_jl": ((-0.5, 1.5),
                                                             "Audio/JL"),
             "go_ls": ((-0.5, 1.5), "Go/LS"), "go_lm": ((-0.5, 1.5), "Go/LM"),
             "go_jl": ((-0.5, 1.5), "Go/JL")}
    task = "SentenceRep"
    root = os.path.expanduser("~/Box/CoganLab")
    layout = get_data(task, root=root)
    folder = 'stats_old'
    mne.set_log_level("ERROR")

    arr = np.arange(24).reshape((2, 3, 4))
    labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
    ad = LabeledArray(arr, labels)
    Labels(['a', 'b', 'c']) @ Labels(['d', 'e', 'f'])

    labels = Labels(np.arange(1000))
    l2d = labels @ labels
    x = l2d.reshape((10, -1)).decompose()
