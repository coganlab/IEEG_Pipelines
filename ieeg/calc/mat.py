import logging
from collections.abc import Iterable, Sequence
import functools
import itertools

import mne

from ieeg.calc.reshape import concatenate_arrays
from ieeg import Signal

import numpy as np
from numpy.matlib import repmat


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
    >>> la # doctest: +ELLIPSIS
    LabeledArray([[[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]],
    <BLANKLINE>
                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]]])
    labels=(('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i')) ... B
    >>> la.to_dict() # doctest: +ELLIPSIS
    {'a': {'c': {'f': 1, 'g': 1, 'h': 1, 'i': 1}, 'd': {'f': 1, 'g': 1,...
    >>> la['a', 'c', 'f'] = 2
    >>> la['a', 'c'] # doctest: +ELLIPSIS
    LabeledArray([2, 1, 1, 1])
    labels=(('f', 'g', 'h', 'i'),) ...

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
        LabeledArray([[[1]]])
        labels=(('a',), ('b',), ('c',)) ...
        >>> data = {'a': {'b': {'c': 1}}, 'd': {'b': {'c': 2, 'e': 3}}}
        >>> LabeledArray.from_dict(data) # doctest: +ELLIPSIS
        LabeledArray([[[ 1., nan]],
        <BLANKLINE>
                      [[ 2.,  3.]]])
        labels=(('a', 'd'), ('b',), ('c', 'e')) ...
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
        >>> bids_root = mne.datasets.epilepsy_ecog.data_path()
        >>> layout = BIDSLayout(bids_root)
        >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
        ... extension=".vhdr", verbose=False)
        Reading 0 ... 269079  =      0.000 ...   269.079 secs...
        >>> LabeledArray.from_signal(raw, dtype=float) # doctest: +ELLIPSIS
        LabeledArray([[-8.98329883e-06,  8.20419238e-06,  7.42294287e-06, ...,
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
        labels=(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'...
               2.69078e+02, 2.69079e+02])) ~201.19 MiB
        >>> epochs = trial_ieeg(raw, "AD1-4, ATT1,2", (-1, 2), preload=True,
        ... verbose=False)
        >>> LabeledArray.from_signal(epochs, dtype=float) # doctest: +ELLIPSIS
        LabeledArray([[[ 0.00021563,  0.00021563,  0.00020703, ...
                        -0.00051445, -0.00050351],
                       [-0.00030586, -0.00030625, -0.00031171, ...
                        -0.00015976, -0.00015664],
                       [-0.00010781, -0.00010469, -0.00010859, ...
                         0.00027695,  0.00030156],
                       ...,
                       [-0.00021483, -0.00021131, -0.00023084, ...
                        -0.00032381, -0.00031444],
                       [-0.00052188, -0.00052852, -0.00053125, ...
                        -0.00047148, -0.00047891],
                       [-0.00033708, -0.00028005, -0.00020934, ...
                        -0.00042341, -0.00040973]]])
        labels=(['AD1-4, ATT1,2'], ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'...
        """

        arr = sig.get_data()
        match sig:
            case mne.io.base.BaseRaw():
                labels = (sig.ch_names, sig.times)
            case mne.BaseEpochs():
                events = list(sig.event_id.keys())
                labels = (events, sig.ch_names, sig.times)
            case mne.evoked.Evoked():
                labels = (sig.ch_names, sig.times)
            case mne.time_frequency.EpochsTFR():
                events = list(sig.event_id.keys())
                labels = (events, sig.ch_names, sig.freqs, sig.times)
            case mne.time_frequency.AverageTFR():
                labels = (sig.ch_names, sig.freqs, sig.times)
            case _:
                raise TypeError(f"Unexpected data type: {type(sig)}")
        return cls(arr, labels, **kwargs)

    def __array_finalize__(self, obj, *args, **kwargs):
        if obj is None:
            return
        self.labels = getattr(obj, 'labels', None)
        super(LabeledArray, self).__array_finalize__(obj, *args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Override numpy ufuncs to preserve labels."""
        labels = self.labels
        inputs = tuple(i.view(np.ndarray) if isinstance(i, LabeledArray)
                       else i for i in inputs)
        if out is not None:
            kwargs['out'] = tuple(o.view(np.ndarray) if
                                  isinstance(o, LabeledArray)
                                  else o for o in out)
        outputs = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if isinstance(outputs, tuple):
            outputs = tuple(LabeledArray(o, labels)
                            if isinstance(o, np.ndarray)
                            else o for o in outputs)
        elif isinstance(outputs, np.ndarray):
            outputs = LabeledArray(outputs, labels)
        return outputs

    @functools.cached_property
    def coord_map(self) -> dict[str, tuple[int, int]]:
        """A dictionary mapping labels to coordinates in the array."""
        return {label: (i, j) for i, labels in enumerate(self.labels)
                for j, label in enumerate(labels)}

    def _str_parse(self, *keys) -> tuple[int, int]:
        for i, key in enumerate(keys):
            match key:
                case list() | tuple():
                    key = list(key)
                    while key:
                        for value in self._str_parse(key.pop(0)):
                            yield value
                case str():
                    yield self.coord_map[key]
                case _:
                    yield i, key

    def _parse_labels(self, keys: tuple) -> tuple:
        new_keys = []
        new_labels = []
        dim = 0
        for key in keys:
            if isinstance(key, str):
                # TODO: fix this is you want multi-label parsing per level
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
        return new_labels, new_keys

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_labels, new_keys = self._parse_labels(keys)
        out = super().__getitem__(tuple(new_keys))
        if isinstance(out, np.ndarray):
            setattr(out, 'labels', tuple(new_labels))
        return out

    def __setitem__(self, key, value):
        coords = []
        if not isinstance(key, tuple):
            key = (key,)
        for i, (dim, num_key) in enumerate(self._str_parse(*key)):
            if isinstance(key[i], str) and key[i] not in self.labels[dim]:
                self.labels[dim] += (key[i],)

            # set num_keys
            while len(coords) <= dim:
                coords.append(None)
            if coords[dim] is None:
                coords[dim] = (num_key,)
            else:
                coords[dim] += (num_key,)
        coords = tuple(c if len(c) > 1 else c[0] for c in coords)
        super(LabeledArray, self).__setitem__(coords, value)

    def __delitem__(self, key):
        dim, num_key = self._str_parse(key)
        if key in self.labels[dim]:
            self.labels = list(self.labels)
            self.labels[dim] = tuple(lab for lab in self.labels[dim]
                                     if lab != key)
            self.labels = tuple(self.labels)
        super(LabeledArray, self).__delitem__(key)

    def __repr__(self):
        size = self.nbytes
        for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
            if size < 1024.0 or unit == 'PiB':
                break
            size /= 1024.0

        return f'{super().__repr__()}\nlabels={self.labels} ~{size:.2f} {unit}'

    def __eq__(self, other):
        if isinstance(other, LabeledArray):
            return np.array_equal(self, other, True) and \
                self.labels == other.labels
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        out = {}
        for k, v in self.items():
            if isinstance(v, LabeledArray):
                out[k] = v.to_dict()
            elif np.isnan(v):
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

    def reshape(self, shape, order='C') -> 'LabeledArray':
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
        >>> ad.reshape((1, 1, 1))
        LabeledArray([[[1]]])
        labels=(('a',), ('b',), ('c',)) ~4.00 B
        >>> arr = np.arange(24).reshape((2, 3, 4))
        >>> labels = (('a', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
        >>> ad = LabeledArray(arr, labels)
        >>> ad.reshape((6, 4)).labels
        (('a-c', 'a-d', 'a-e', 'b-c', 'b-d', 'b-e'), ('f', 'g', 'h', 'i'))
        >>> ad.reshape((6, 4), 'F').labels
        (('a-c', 'b-c', 'a-d', 'b-d', 'a-e', 'b-e'), ('f', 'g', 'h', 'i'))
        >>> ad.reshape((2, 12)).labels # doctest: +ELLIPSIS
        (('a', 'b'), ('c-f', 'c-g', 'c-h', 'c-i', 'd-f', 'd-g', 'd-h', 'd-i'...
        """
        new_array = super().reshape(*shape, order=order)
        new_labels = label_reshape(self.labels, shape, order)
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
        LabeledArray([[[1]]])
        labels=(('a',), ('pre-b',), ('c',)) ...
        """
        assert 0 <= level < self.ndim, "level must be >= 0 and < ndim"
        labels = list(self.labels)
        labels[level] = tuple(pre + lab for lab in labels[level])
        self.labels = tuple(labels)
        return LabeledArray(self.view(np.ndarray), self.labels)

    def combine(self, levels: tuple[int, int],
                delim: str = '-', drop_nan: bool = True) -> 'LabeledArray':
        """Combine any levels of a LabeledArray into the lower level

        Takes the input LabeledArray and rearranges its dimensions.

        Parameters
        ----------
        levels : tuple[int, int]
            The levels to combine, e.g. (0, 1) will combine the 1st and 2nd
            level of the array labels into one level at the 2nd level.
        delim : str, optional
            The delimiter to use when combining labels, by default '-'
        drop_nan : bool, optional
            Whether to drop all NaN columns, by default True

        Returns
        -------
        LabeledArray
            The combined LabeledArray

        Examples
        --------
        >>> data = {'a': {'b': {'c': 1}}}
        >>> ad = LabeledArray.from_dict(data, dtype=int)
        >>> ad.combine((0, 2)) # doctest: +ELLIPSIS
        LabeledArray([[1]])
        labels=(('b',), ('a-c',)) ...
        """

        assert levels[0] >= 0, "first level must be >= 0"
        assert levels[1] > levels[0], "second level must be > first level"

        new_labels = list(self.labels)

        new_labels.pop(levels[0])

        new_labels[levels[1] - 1] = tuple(
            f'{i}{delim}{j}' for i in
            self.labels[levels[0]] for j in self.labels[levels[1]])

        new_shape = list(self.shape)

        new_shape[levels[1]] = self.shape[levels[0]] * self.shape[
            levels[1]]

        new_shape.pop(levels[0])

        new_array = np.reshape(self, new_shape)

        if drop_nan:
            return LabeledArray(new_array, new_labels).dropna()
        else:
            return LabeledArray(new_array, new_labels)

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
        >>> ad.dropna() # doctest: +ELLIPSIS
        LabeledArray([[[1.]]])
        labels=(('a',), ('b',), ('c',)) ...
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
        new_array = LabeledArray(np.array(self)[index], new_labels)
        return new_array


def label_reshape(labels: tuple[tuple[str, ...], ...], shape: tuple[int, ...],
                  order: str = 'C', delim: str = '-'
                  ) -> tuple[tuple[str, ...], ...]:
    """Reshape the labels of a LabeledArray.

    Takes the labels corresponding to the shape of an array and reshapes them
    into the new shape. This is accomplished by 'flattening' the labels and
    then reassigning and reducing them in the new shape.

    Parameters
    ----------
    labels : tuple[tuple[str, ...], ...]
        The labels to reshape.
    shape : tuple[int, ...]
        The new shape of the labels.
    order : str, optional
        The order to reshape the labels in, by default 'C'
    delim : str, optional
        The delimiter to use when combining labels, by default '-'

    Returns
    -------
    tuple[tuple[str, ...], ...]
        The reshaped labels.

    Examples
    --------
    >>> labels = (('az', 'b'), ('c', 'd', 'e'), ('f', 'g', 'h', 'i'))
    >>> label_reshape(labels, (6, 4))
    (('az-c', 'az-d', 'az-e', 'b-c', 'b-d', 'b-e'), ('f', 'g', 'h', 'i'))
    >>> label_reshape(labels, (6, 4), 'F')
    (('az-c', 'b-c', 'az-d', 'b-d', 'az-e', 'b-e'), ('f', 'g', 'h', 'i'))
    >>> label_reshape(labels, (2, 12)) # doctest: +ELLIPSIS
    (('az', 'b'), ('c-f', 'c-g', 'c-h', 'c-i', 'd-f', 'd-g', 'd-h', 'd-i'...
    >>> label_reshape(labels, (3, 2, 4))
    (('az', 'az-b', 'b'), ('c-d-e', 'c-d-e'), ('f', 'g', 'h', 'i'))
    """
    if order == 'F':
        prod = map(reversed, itertools.product(*reversed(labels)))
    else:
        prod = itertools.product(*labels)
    flattened = tuple(map(delim.join, prod))
    temp = np.full(shape, '0'*max(map(len, flattened)))
    if order == 'F':
        temp.T.flat = flattened
    else:
        temp.flat = flattened

    # now that temp is a char array of corresponding labels, we can factor the
    # matrix into the new labels
    new_labels = [[] for _ in range(len(shape))]
    for i, dim in enumerate(shape):
        if len(labels[i]) == dim == 1:
            new_labels[i] = labels[i]
            continue
        for j in range(dim):
            row = list(np.take(temp, j, axis=i).flat)
            delimed = tuple(map(lambda x: tuple(x.split(delim)), row))
            common = _longest_common_substring(delimed)
            if len(common) == 0:
                logging.warn(f"Could not find common substring for "
                             f"labels dimension {i} index {j}")
                common = sorted(list(set(d[i] for d in delimed)),
                                key=labels[i].index)
            new_labels[i].append(delim.join(common))
    return tuple(map(tuple, new_labels))


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
        data = np.atleast_1d(data)
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
        arr = (inner_array(d) for d in data.values())
        arr = [a for a in arr if a is not None]
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


def inner_dict(data: np.ndarray) -> dict | None:
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
    >>> inner_dict(data)
    {0: {0: {0: 1}}}
    >>> data = np.array([[[1, np.nan]],
    ...                  [[2, 3]]])
    >>> inner_dict(data)
    {0: {0: {0: 1.0, 1: nan}}, 1: {0: {0: 2.0, 1: 3.0}}}
    """
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


if __name__ == "__main__":
    ad = LabeledArray([[[1, 2]]], labels=(('a',), ('b',), ('c', 'd')))
    # import os
    # from ieeg.io import get_data
    # from utils.mat_load import load_dict
    # import mne
    # ins, axis, exp = ([np.array([]), np.array([[1., 2.], [3., 4.]]),
    #                    np.array([[5., 6., 7.], [8., 9., 10.]])], 0,
    #                   np.array([[1, 2, np.nan], [3, 4, np.nan],
    #                             [5, 6, 7], [8, 9, 10]]))
    # outs = concatenate_arrays(ins, axis)
    # ar = LabeledArray.from_dict(dict(a=ins[1], b=ins[2]))
    # x = ar["a"]
    # y = ar.to_dict()
    # conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
    #          "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
    #          "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
    #          "go_jl": (-0.5, 1.5)}
    # task = "SentenceRep"
    # root = os.path.expanduser("~/Box/CoganLab")
    # layout = get_data(task, root=root)
    #
    # mne.set_log_level("ERROR")

    # data = LabeledArray.from_dict(dict(
    #     power=load_dict(layout, conds, "power", False),
    #     zscore=load_dict(layout, conds, "zscore", False)))

    # dict_data = dict(
    #     power=load_dict(layout, conds, "power", False))
    # # zscore=load_dict(layout, conds, "zscore", False))
    #
    # keys = inner_all_keys(dict_data)
    #
    # data = LabeledArray.from_dict(dict_data)
    # data.__repr__()

    # data = SparseArray(dict_data)

    # power = data["power"]
