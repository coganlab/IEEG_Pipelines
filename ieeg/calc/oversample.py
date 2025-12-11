from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold

import itertools
from functools import partial
from ieeg.calc.fast import mixup, norm, mixup2
from ieeg.arrays.api import array_namespace, is_numpy, intersect1d, setdiff1d, is_torch
from decimal import Decimal

Array2D = NDArray[Tuple[Literal[2], ...]]
Vector = NDArray[Literal[1]]


class MinimumNaNSplit(RepeatedStratifiedKFold):
    """A Repeated Stratified KFold iterator that splits the data into sections

    This class splits the data into sections, checking that the training set
    never has fewer than the specified number of non-NaN values.
    Parameters
    ----------
    n_splits : int
        The number of splits.
    n_repeats : int, optional
        The number of times to repeat the splits, by default 10.
    random_state : int, optional
        The random state to use, by default None.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.vstack((np.arange(1, 13).reshape(6, 2), np.full((6, 2), np.nan)))
    >>> y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    >>> msn = MinimumNaNSplit(5, 10, min_non_nan=2)
    >>> for train, test in msn.split(X, y):
    ...     print("train:", train, X[train, 0], y[train], "test:", test)
    train: [2 3 4 5] test: [0 1 6 7]
    train: [0 1 6 7] test: [2 3 4 5]
    train: [2 3 4 5] test: [0 1 6 7]
    train: [0 1 6 7] test: [2 3 4 5]
    train: [2 3 4 5] test: [0 1 6 7]
    train: [0 1 6 7] test: [2 3 4 5]
    >>> msn = MinimumNaNSplit(2, 3, which='test', min_non_nan=1)
    >>> for train, test in msn.split(X, y):
    ...     print("train:", train, "test:", test, X[test, 0], y[test])
    train: [1 3 4 7] test: [0 2 5 6]
    train: [0 2 5 6] test: [1 3 4 7]
    train: [0 3 5 7] test: [1 2 4 6]
    train: [1 2 4 6] test: [0 3 5 7]
    train: [1 2 5 6] test: [0 3 4 7]
    train: [0 3 4 7] test: [1 2 5 6]
    """

    def __init__(self, n_splits: int, n_repeats: int = 10,
                 random_state: int = None, min_non_nan: int = 2,
                 which: str = 'train'):
        super(MinimumNaNSplit, self).__init__(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits
        self.min_non_nan = min_non_nan
        if which not in ('train', 'test'):
            raise ValueError("which must be either 'train' or 'test'")
        self.which = which

    def split(self, X, y=None, groups=None):

        xp = array_namespace(*[a for a in (X, y, groups) if a is not None])

        # find where the nans are
        where = xp.isnan(X).any(axis=tuple(range(X.ndim))[1:])
        not_where = xp.nonzero(~where)[0]
        where = xp.nonzero(where)[0]

        splits = self._splits(X, y, groups, xp)

        # if there are no nans, then just split the data
        if len(where) == 0:
            yield from splits
            return
        elif (n_non_nan := not_where.shape[0]) < (n_min := self.min_non_nan +
                                                  1):
            raise ValueError(f"Need at least {n_min} non-nan values, but only"
                             f" have {n_non_nan}")

        check = {'train': lambda t: setdiff1d(not_where, t, xp=xp,
                                              assume_unique=True),
                 'test': lambda t: intersect1d(not_where, t, xp=xp,
                                               assume_unique=True)}

        # check that all training sets for each kfold within each repetition
        # have at least min_non_nan non-nan values
        idxs = [xp.nonzero(i == y)[0] for i in xp.unique(y)]
        kfold_set = [None] * self.n_splits
        while element := next(splits, False):
            for i in range(self.n_splits):
                if i == 0:
                    train, test = element
                else:
                    train, test = next(splits)

                # if any test set has more non-nan values than the total number
                # of non-nan values minus the minimum number of non-nan values,
                # then throw out the split and append an extra repetition
                if all(intersect1d(check[self.which](test), i, xp=xp,
                                   assume_unique=True).shape[0] <
                       self.min_non_nan for i in idxs):
                    for _ in range(i + 1, self.n_splits):
                        next(splits)
                    extra = self._splits(X, y, groups, xp)
                    one_rep = itertools.islice(extra, self.n_splits)
                    splits = itertools.chain(one_rep, splits)
                    break
                kfold_set[i] = (train, test)
            else:
                yield from kfold_set

    def _splits(self, X, y, groups, xp):
        splits = super(MinimumNaNSplit, self).split(X, y, groups)
        if is_numpy(xp):
            return splits
        # For array-API backends ensure indices live on same device as X
        try:
            device = X.device  # torch or cupy arrays expose device
        except Exception:
            device = None
        if is_torch(xp):
            import torch
            def to_backend(arr):
                return torch.as_tensor(arr, dtype=torch.long, device=device)
        else:
            # CuPy or other backends
            def to_backend(arr):
                return xp.asarray(arr)
        return ( (to_backend(train), to_backend(test)) for train, test in splits )

    @staticmethod
    def oversample(arr: np.ndarray, func: callable = mixup,
                   axis: int = 1, copy: bool = True, seed=None) -> np.ndarray:
        """Oversample nan rows using func

        Parameters
        ----------
        arr : array
            The data to oversample.
        func : callable
            The function to use to oversample the data.
        axis : int
            The axis along which to apply func.
        copy : bool
            Whether to copy the data before oversampling.

        Examples
        --------
        >>> np.random.seed(0)
        >>> arr = np.array([[1, 2], [4, 5], [7, 8],
        ... [float("nan"), float("nan")]])
        >>> MinimumNaNSplit.oversample(arr, norm, 0)
        array([[1.        , 2.        ],
               [4.        , 5.        ],
               [7.        , 8.        ],
               [8.32102813, 5.98018098]])
        >>> MinimumNaNSplit.oversample(arr, mixup, 0, seed=42) # doctest: +SKIP
        array([[1.        , 2.        ],
               [4.        , 5.        ],
               [7.        , 8.        ],
               [5.24946679, 6.24946679]])
        """
        if copy:
            arr = arr.copy()

        axis = arr.ndim + axis if axis < 0 else axis

        if arr.ndim <= 0:
            raise ValueError("Cannot apply func to a 0-dimensional array")
        elif seed is not None:
            func(arr, axis, seed=seed)
        else:
            func(arr, axis)
        return arr

    def shuffle_labels(self, arr: np.ndarray, labels: np.ndarray,
                       trials_ax: int = 0, min_trials: int = 1):
        """Shuffle the labels while making sure that the minimum non nan
        trials are kept

        Parameters
        ----------
        arr : array
            The data to shuffle.
        labels : array
            The labels to shuffle.
        trials_ax : int
            The axis along which to apply func.
        min_trials : int
            The minimum number of non-nan trials to keep. By default,
            self.n_splits

        Examples
        --------
        >>> np.random.seed(0)
        >>> arr = np.array([[[1, 2], [4, 5], [7, 8],
        ... [float("nan"), float("nan")]]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> MinimumNaNSplit(1).shuffle_labels(arr, labels, 1, 1)
        >>> labels
        array([1, 1, 0, 0])
        """
        xp = array_namespace(arr)
        cats = xp.unique(labels)
        gt_labels = [0] * cats.shape[0]
        min_trials *= self.n_splits
        i = 0
        while not all(g >= min_trials for g in gt_labels):
            if is_torch(xp):
                import torch
                perm = torch.randperm(labels.shape[0], device=labels.device)
                labels[:] = labels[perm]
            else:
                xp.random.shuffle(labels)
            for j, l in enumerate(cats):
                if is_torch(xp):
                    import torch
                    idx = torch.nonzero(labels == l, as_tuple=False).flatten()
                    eval_arr = torch.index_select(arr, dim=trials_ax, index=idx)
                else:
                    idx = xp.flatnonzero(labels == l)
                    eval_arr = xp.take(arr, idx, trials_ax)
                gt_labels[j] = xp.min(xp.sum(
                    xp.all(~xp.isnan(eval_arr), axis=2), axis=trials_ax))
            if sum(gt_labels) < min_trials * cats.shape[0]:
                raise ValueError("Not enough non-nan trials to shuffle")
            i += 1
            if i > 100000:
                raise ValueError("Could not find a shuffle that satisfies the"
                                 " minimum number of non-nan trials "
                                 f"{gt_labels}")


def oversample_nan(arr: np.ndarray, func: callable, axis: int = 1,
                   copy: bool = True, seed: int = None) -> np.ndarray:
    """Oversample nan rows using func

    Parameters
    ----------
    arr : array
        The data to oversample.
    func : callable
        The function to use to oversample the data.
    axis : int
        The axis along which to apply func.
    copy : bool
        Whether to copy the data before oversampling.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> oversample_nan(arr, norm, 0)
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [8.32102813, 5.98018098]])
    >>> oversample_nan(arr, mixup, 0, seed=42) # doctest: +SKIP
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.24946679, 6.24946679]])
    >>> arr3 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr3[0, 2, :] = [float("nan")] * 4
    >>> oversample_nan(arr3, mixup, 1, seed=42) # doctest: +SKIP
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 2.33404428,  3.33404428,  4.33404428,  5.33404428]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> oversample_nan(arr3, norm, 1)
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 3.95747597,  7.4817864 ,  7.73511598,  3.04544424]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    """

    if copy:
        arr = arr.copy()

    axis = arr.ndim + axis if axis < 0 else axis

    if arr.ndim <= 0:
        raise ValueError("Cannot apply func to a 0-dimensional array")
    elif func is mixup and seed is not None:
        func(arr, axis, seed=seed)
    else:
        func(arr, axis)
    return arr


def find_nan_indices(arr: np.ndarray, obs_axis: int) -> tuple:
    """Find the indices of rows with and without NaN values

    Parameters
    ----------

    arr : array
        The data to find indices.
    obs_axis : int
        The axis along which to apply func.

    Returns
    -------
    tuple
        A tuple of two arrays containing the indices of rows with and without
        NaN values.

    Examples
    --------
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> find_nan_indices(arr, 0) # doctest: +ELLIPSIS
    (array([3]... array([0, 1, 2]...)
    """

    # Initialize boolean mask of rows with NaN values
    not_obs = tuple(i for i in range(arr.ndim) if i != obs_axis)

    # Check each row individually
    nan = np.any(np.isnan(arr), axis=not_obs)

    # Get indices of rows with and without NaN values using boolean indexing
    nan_rows = np.flatnonzero(nan)
    non_nan_rows = np.flatnonzero(~nan)

    return nan_rows, non_nan_rows


def sortbased_rand(n_range: int, iterations: int, n_picks: int = -1):
    """Generate random numbers using sort-based sampling, resulting in a random
    choice generation without replacement along the first axis.

    Parameters
    ----------
    n_range : int
        The range of numbers to sample from.
    iterations : int
        The number of iterations to run.
    n_picks : int
        The number of numbers to pick from the range. If -1, then the number of
        picks is equal to the range.

    Returns
    -------
    array
        An array of shape (iterations, n_picks) containing the random numbers.

    References
    ----------
    [1] `stackoverflow link <https://stackoverflow.com/questions/31955660/effic
    iently-generating-multiple-instances-of-numpy-random-choice-without-replace
    /31958263#31958263>`_

    Examples
    --------
    >>> np.random.seed(0)
    >>> sortbased_rand(10, 5, 3)
    array([[9, 4, 6],
           [6, 4, 5],
           [4, 6, 9],
           [4, 0, 2],
           [3, 7, 6]])
    """
    return np.argsort(np.random.rand(iterations, n_range), axis=1
                      )[:, :n_picks]


# def mixup2(arr: Array, labels: Array, obs_axis: int,
#            alpha: float = 1., seed: int = None, _isnan=None,
#            _time_ax: int = -1) -> None:
#     """Mixup the data using the labels
#
#     Parameters
#     ----------
#     arr : array
#         The data to mixup.
#     labels : array
#         The labels to use for mixing.
#     obs_axis : int
#         The axis along which to apply func.
#     alpha : float
#         The alpha value for the beta distribution.
#     seed : int
#         The seed for the random number generator.
#
#     Examples
#     --------
#     >>> np.random.seed(0)
#     >>> arr = np.array([[1, 2], [4, 5], [7, 8],
#     ... [float("nan"), float("nan")]])
#     >>> labels = np.array([0, 0, 1, 1])
#     >>> mixup2(arr[None], labels, 1)
#     >>> arr
#     array([[1.        , 2.        ],
#            [4.        , 5.        ],
#            [7.        , 8.        ],
#            [6.03943491, 7.03943491]])
#     >>> arr3 = np.arange(24, dtype=float).reshape(2,3,4)
#     >>> arr3[:, 2, :] = float("nan")
#     >>> mixup2(arr3, np.array([1, 0, 1]), 1, seed=42)
#     >>> arr3
#     array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
#             [ 4.        ,  5.        ,  6.        ,  7.        ],
#             [ 1.99984539,  2.99984539,  3.99984539,  4.99984539]],
#     <BLANKLINE>
#            [[12.        , 13.        , 14.        , 15.        ],
#             [16.        , 17.        , 18.        , 19.        ],
#             [12.25137354, 13.25137354, 14.25137354, 15.25137354]]])
#     >>> import cupy as cp
#     >>> arr4 = cp.array([[1, 2],[3,4], [5, 6],
#     ... [7, 8], [9, 10], [cp.nan, cp.nan]])
#     >>> labels4 = cp.array([0, 0, 0, 1, 1, 1, 1])
#     >>> mixup2(arr4, labels4, 0, seed=0)
#     >>> arr4
#     array([[1.        , 2.        ],
#            [4.        , 5.        ],
#            [7.        , 8.        ],
#            [4.54459201, 5.54459201]])
#     >>> arr4 = cp.array([[1, 2, 3, 4],[3,4, 5, 6], [5, 6, 7, 8],
#     ... [7, 8, 9, 10], [9, 10, 11, 12], [cp.nan, cp.nan, cp.nan, cp.nan]])
#     >>> new = cp.stack(cp.stack(tuple(arr4 for _ in range(1000))) for _ in range(1000))
#     >>> mixup2(new, labels4, 2, seed=0)
#     >>> new[0, 0]
#     array([[ 1.        ,  2.        ,  3.        ,  4.        ],
#            [ 3.        ,  4.        ,  5.        ,  6.        ],
#            [ 5.        ,  6.        ,  7.        ,  8.        ],
#            [ 7.        ,  8.        ,  9.        , 10.        ],
#            [ 9.        , 10.        , 11.        , 12.        ],
#            [ 6.37764196,  7.37764196,  8.37764196,  9.37764196]])
#     >>> new = cp.ascontiguousarray(cp.stack(cp.stack(tuple(arr4 for _ in range(1000))) for _ in range(1000)).swapaxes(1,2))
#     >>> list(mixup2(new.copy(), labels4, 1, seed=0) for _ in range(100))
#     >>> mixup2(new, labels4, 1, seed=0)
#     >>> new[0, -1, 0:10]
#     array([[ 1.        ,  2.        ,  3.        ,  4.        ],
#            [ 3.        ,  4.        ,  5.        ,  6.        ],
#            [ 5.        ,  6.        ,  7.        ,  8.        ],
#            [ 7.        ,  8.        ,  9.        , 10.        ],
#            [ 9.        , 10.        , 11.        , 12.        ],
#            [ 6.37764196,  7.37764196,  8.37764196,  9.37764196]])
#            """
#
#     xp = array_namespace(arr, labels)
#
#     if seed is not None:
#         xp.random.seed(seed)
#
#     arr_moved = xp.moveaxis(arr, [obs_axis, _time_ax], [-2, -1])
#
#     if _isnan is None:
#         isnan = xp.isnan(arr_moved).any(-1)
#     else:
#         isnan = _isnan.swapaxes(obs_axis, -1)
#
#     # get any and all in single call
#     isn_sum = isnan.sum(-1, dtype='uintp')
#     if (isn_sum == isnan.shape[-1]).any():
#         raise ValueError("Cannot mixup if any rows are completely NaN")
#
#     unique_labels, label_ids = xp.unique(labels, return_inverse=True)
#
#     mask = isn_sum > 0
#     # Precompute nan and non-nan indices for all batches in one pass
#     batch_shape = isnan.shape[:-1]
#     B = int(xp.prod(xp.asarray(batch_shape)))
#     obs = isnan.shape[-1]
#     mask_flat = mask.reshape(B)
#     isnan_flat = isnan.reshape(B, obs)
#
#     proc_flat = xp.flatnonzero(mask_flat)
#     if proc_flat.size == 0:
#         return
#     selected = isnan_flat[proc_flat]
#
#     # Partition flattened mask once to separate non-NaN (False) and NaN (True)
#     total_elems = selected.size
#     num_non_nan = int((~selected).sum())
#     flat_mask = selected.ravel().astype(bool)
#     if num_non_nan == 0:
#         nn_flat = flat_mask[:0]
#         n_flat = xp.arange(total_elems)
#     elif num_non_nan == total_elems:
#         nn_flat = xp.arange(total_elems)
#         n_flat = flat_mask[:0]
#     else:
#         flat_order = xp.argpartition(flat_mask, num_non_nan - 1)
#         nn_flat = flat_order[:num_non_nan]
#         n_flat = flat_order[num_non_nan:]
#
#     # Recover per-batch row indices and observation indices
#     rows_nn, cols_nn = divmod(nn_flat, obs)
#     rows_n, cols_n = divmod(n_flat, obs)
#
#     # Per-batch counts and offsets (for choices2 sampling)
#     counts_nn = xp.bincount(rows_nn, minlength=proc_flat.shape[0])
#     counts_n = xp.bincount(rows_n, minlength=proc_flat.shape[0])
#     offsets_nn = xp.empty(counts_nn.shape[0] + 1, dtype=counts_nn.dtype)
#     offsets_nn[0] = 0
#     offsets_nn[1:] = xp.cumsum(counts_nn)
#
#     # Sanity: any batch with nans must have at least one non-nan candidate
#     if xp.any((counts_n > 0) & (counts_nn == 0)):
#         raise ValueError("Not enough non-nan values to mixup")
#
#     # Precompute choices2 for all targets at once using offsets into cols_nn
#     if rows_n.size:
#         pool_sizes = counts_nn[rows_n]
#         r2_all = (xp.random.random(rows_n.shape[0]) * pool_sizes).astype(offsets_nn.dtype)
#         pos = offsets_nn[rows_n] + r2_all
#         choices2_all = cols_nn[pos]
#     else:
#         choices2_all = rows_n  # empty
#
#     # Build first-choice (same-class, same-batch) indices for all targets at once
#     K = int(unique_labels.shape[0])
#     target_class_ids = label_ids[cols_n]
#     non_class_ids = label_ids[cols_nn]
#     comp = rows_nn * K + non_class_ids
#     max_len = int(proc_flat.shape[0]) * K
#     counts_comp = xp.bincount(comp, minlength=max_len)
#     offsets_comp = xp.empty(counts_comp.shape[0] + 1, dtype=counts_comp.dtype)
#     offsets_comp[0] = 0
#     offsets_comp[1:] = xp.cumsum(counts_comp)
#     order_comp = xp.argsort(comp)
#     cols_nn_sorted = cols_nn[order_comp]
#     comp_targets = rows_n * K + target_class_ids
#     pool_sizes1 = counts_comp[comp_targets]
#     if xp.any(pool_sizes1 == 0):
#         raise ValueError("Not enough non-nan values to mixup")
#     r1_all = (xp.random.random(rows_n.shape[0]) * pool_sizes1).astype(offsets_comp.dtype)
#     pos1 = offsets_comp[comp_targets] + r1_all
#     choices1_all = cols_nn_sorted[pos1]
#
#     # Mixup coefficients for all targets
#     lam_all = xp.random.beta(alpha, alpha, size=rows_n.shape[0]).astype(arr.dtype)
#     xp.maximum(lam_all, 1 - lam_all, out=lam_all)
#
#     # Vectorized in-place assignment using multi-indexing (no reshape)
#     batch_coords = xp.unravel_index(proc_flat, batch_shape)
#     batch_index_tuple = tuple(coord[rows_n] for coord in batch_coords)
#     lam_all = lam_all[:, None]  # for broadcasting
#     arr_moved[batch_index_tuple + (cols_n,)] = \
#         lam_all * arr_moved[batch_index_tuple + (choices1_all,)]
#     xp.subtract(1, lam_all, out=lam_all)
#     arr_moved[batch_index_tuple + (cols_n,)] += \
#         lam_all * arr_moved[batch_index_tuple + (choices2_all,)]


def resample(arr: np.ndarray, sfreq: int | float | list[int | float], new_sfreq: int | float | list[int | float],
             axis: int = -1) -> np.ndarray:
    """Resample an array through linear interpolation.

    Parameters
    ----------
    arr : array
        The array to resample.
    sfreq : int
        The original sampling frequency.
    new_sfreq : int
        The new sampling frequency.

    Returns
    -------
    resampled : array
        The resampled array.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> resample(arr, 10, 5)
    array([0.  , 2.25, 4.5 , 6.75, 9.  ])
    >>> resample(arr, 10.2, 5.1)
    array([0.  , 2.25, 4.5 , 6.75, 9.  ])
    >>> resample(arr, 10, 20)
    array([0.        , 0.47368421, 0.94736842, 1.42105263, 1.89473684,
           2.36842105, 2.84210526, 3.31578947, 3.78947368, 4.26315789,
           4.73684211, 5.21052632, 5.68421053, 6.15789474, 6.63157895,
           7.10526316, 7.57894737, 8.05263158, 8.52631579, 9.        ])
    >>> resample(arr, 10, 7)
    array([0. , 1.5, 3. , 4.5, 6. , 7.5, 9. ])
    >>> arr = np.arange(30).reshape(5,6)
    >>> resample(arr, 6, 10)
    array([[ 0.        ,  0.55555556,  1.11111111,  1.66666667,  2.22222222,
             2.77777778,  3.33333333,  3.88888889,  4.44444444,  5.        ],
           [ 6.        ,  6.55555556,  7.11111111,  7.66666667,  8.22222222,
             8.77777778,  9.33333333,  9.88888889, 10.44444444, 11.        ],
           [12.        , 12.55555556, 13.11111111, 13.66666667, 14.22222222,
            14.77777778, 15.33333333, 15.88888889, 16.44444444, 17.        ],
           [18.        , 18.55555556, 19.11111111, 19.66666667, 20.22222222,
            20.77777778, 21.33333333, 21.88888889, 22.44444444, 23.        ],
           [24.        , 24.55555556, 25.11111111, 25.66666667, 26.22222222,
            26.77777778, 27.33333333, 27.88888889, 28.44444444, 29.        ]])
    >>> ind1 = [0, 1, 2, 3, 4, 5]
    >>> ind2 = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
    >>> resample(arr, ind1, ind2, axis=1)
    array([[ 0. ,  0.5,  1.5,  2.5,  3.5,  4.5],
           [ 6. ,  6.5,  7.5,  8.5,  9.5, 10.5],
           [12. , 12.5, 13.5, 14.5, 15.5, 16.5],
           [18. , 18.5, 19.5, 20.5, 21.5, 22.5],
           [24. , 24.5, 25.5, 26.5, 27.5, 28.5]])
    >>> ind2 = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 10]
    >>> resample(arr, ind1, ind2, axis=1)
    array([[ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8,  5. ],
           [ 6. ,  6.2,  6.4,  6.6,  6.8,  7. ,  7.2,  7.4,  7.6,  7.8, 11. ],
           [12. , 12.2, 12.4, 12.6, 12.8, 13. , 13.2, 13.4, 13.6, 13.8, 17. ],
           [18. , 18.2, 18.4, 18.6, 18.8, 19. , 19.2, 19.4, 19.6, 19.8, 23. ],
           [24. , 24.2, 24.4, 24.6, 24.8, 25. , 25.2, 25.4, 25.6, 25.8, 29. ]])

    >>> xp = [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> np.random.seed(100)
    >>> x = np.random.rand(10)
    >>> fp = np.random.rand(10, 5, 5)
    >>> resample(fp, xp, x)[0, :, 0]
    array([0.17196795, 0.2838014 , 0.73404496, 0.34674289, 0.1532049 ])
    >>> resample(fp, xp, x, 1)[0, 0]
    array([0.42148282, 0.7778097 , 0.71951549, 0.41589901, 0.1476043 ])
    """
    while axis < 0:
        axis += arr.ndim
    if not np.isscalar(sfreq):
        assert not np.isscalar(new_sfreq), ValueError(
            "If sfreq is a list, new_sfreq must also be a list")
        o_indices = np.asarray(sfreq)
        indices = np.asarray(new_sfreq)
        new_samps = len(new_sfreq)
        # make sure all indices are sorted and unique
        assert np.all(np.diff(o_indices) > 0), ValueError(
            f"sfreq:\n {sfreq}\n is not sorted and unique")
    elif not np.isscalar(new_sfreq):
        raise ValueError("If new_sfreq is a list, sfreq must also be a list")
    elif sfreq == new_sfreq:
        return arr
    elif not sfreq % 1 == 0:
        num, denom = Decimal(str(sfreq)).as_integer_ratio()
        return resample(arr, num, new_sfreq * denom, axis)
    elif not new_sfreq % 1 == 0:
        num, denom = Decimal(str(new_sfreq)).as_integer_ratio()
        return resample(arr, sfreq * denom, num, axis)
    else:
        # Directly calculate the new sample points
        seconds = arr.shape[axis] / sfreq
        o_indices = np.arange(arr.shape[axis])
        new_samps = int(round(new_sfreq * seconds))
        indices = np.linspace(0, arr.shape[axis] - 1, new_samps)

    if arr.ndim == 1:
        return np.interp(indices, o_indices, arr)

    # Pre-allocate output in final shape to avoid reshape/swapaxes copies
    # Output shape: replace axis dimension with new_samps
    out_shape = tuple(arr.shape[i] if i != axis else new_samps 
                      for i in range(arr.ndim))
    out = np.empty(out_shape, dtype=f"f{arr.dtype.itemsize}")
    
    # Move axis to end for easier indexing (creates view, no copy)
    arr_moved = np.moveaxis(arr, axis, -1)  # shape: (..., n_old)
    out_moved = np.moveaxis(out, axis, -1)  # shape: (..., n_new)
    
    n_old = arr.shape[axis]
    
    # Find left indices for interpolation (vectorized)
    # Use side='right' to match np.interp behavior
    j = np.searchsorted(o_indices, indices, side='right') - 1
    
    # Handle boundary cases like np.interp: copy edge values
    # Values < o_min: j will be -1, clamp to 0 and set d=0 (use first value)
    # Values >= o_max: j will be len(o_indices)-1, clamp to len-2 and set d=1 (use last value)
    j_low = j < 0
    j_high = j >= n_old - 1
    j = np.clip(j, 0, n_old - 2)
    
    # Compute interpolation weights (vectorized)
    # Pre-compute differences for efficiency
    # o_indices is guaranteed sorted and unique by assertion, so o_diff > 0
    o_diff = o_indices[1:] - o_indices[:-1]
    denom = o_diff[j]
    d = (indices - o_indices[j]) / denom
    
    # Handle boundary cases: copy edge values (like np.interp)
    d[j_low] = 0.0  # Use first value for indices < o_min
    d[j_high] = 1.0  # Use last value for indices >= o_max
    # Clamp d to [0, 1] for safety
    d = np.clip(d, 0, 1)

    # Optimized interpolation with improved memory locality using in-place operations
    # arr_moved shape: (d0, d1, ..., dn, n_old) - view from moveaxis (no copy)
    # out_moved shape: (d0, d1, ..., dn, n_new) - view of output (no copy)
    # j shape: (n_new,)
    # d shape: (n_new,)
    
    # Extract values along last axis using advanced indexing
    # arr_moved[..., j] selects j[k] along last axis for each k, broadcasting across leading dims
    # Result shape: (d0, d1, ..., dn, n_new)
    arr_j = arr_moved[..., j]  # shape: (..., n_new)
    arr_jp1 = arr_moved[..., j + 1]  # shape: (..., n_new)
    
    # Compute interpolation using in-place operations to improve locality
    # Formula: out = arr_j + d*(arr_jp1 - arr_j)
    # Step 1: Compute difference directly into output (reuse output buffer)
    # np.subtract(arr_jp1, arr_j, out=out_moved)
    out_moved[...] = arr_jp1 - arr_j
    # Step 2: Multiply by interpolation weight in-place
    # np.multiply(out_moved, d, out=out_moved)
    out_moved *= d
    # Step 3: Add base value in-place
    # np.add(arr_j, out_moved, out=out_moved)
    out_moved += arr_j

    return out

def multiInterp2d(x, xp, fp):
    i = np.arange(x.size)
    j = np.searchsorted(xp, x) - 1
    d = (x - xp[j]) / (xp[j + 1] - xp[j])
    return (1 - d) * fp[i, j] + fp[i, j + 1] * d
