from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold

import itertools
from functools import partial
from ieeg.calc.fast import mixup, norm

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
    >>> X = np.vstack((np.arange(1, 9).reshape(4, 2), np.full((4, 2), np.nan)))
    >>> y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> msn = MinimumNaNSplit(2, 3)
    >>> for train, test in msn.split(X, y):
    ...     print("train:", train, "test:", test)
    train: [1 3 4 6] test: [0 2 5 7]
    train: [0 2 5 7] test: [1 3 4 6]
    train: [1 3 4 7] test: [0 2 5 6]
    train: [0 2 5 6] test: [1 3 4 7]
    train: [0 3 5 7] test: [1 2 4 6]
    train: [1 2 4 6] test: [0 3 5 7]
    >>> msn = MinimumNaNSplit(2, 3, which='test', min_non_nan=1)
    >>> for train, test in msn.split(X, y):
    ...     print("train:", train, "test:", test)
    train: [2 3 4 5] test: [0 1 6 7]
    train: [0 1 6 7] test: [2 3 4 5]
    train: [0 1 2 7] test: [3 4 5 6]
    train: [3 4 5 6] test: [0 1 2 7]
    train: [3 4 5 7] test: [0 1 2 6]
    train: [0 1 2 6] test: [3 4 5 7]
    """

    def __init__(self, n_splits: int, n_repeats: int = 10,
                 random_state: int = None, min_non_nan: int = 2,
                 which: str = 'train'):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats,
                         random_state=random_state)
        self.n_splits = n_splits
        self.min_non_nan = min_non_nan
        if which not in ('train', 'test'):
            raise ValueError("which must be either 'train' or 'test'")
        self.which = which

    def split(self, X, y=None, groups=None):

        # find where the nans are
        where = np.isnan(X).any(axis=tuple(range(X.ndim))[1:])
        not_where = np.where(~where)[0]
        where = np.where(where)[0]

        # if there are no nans, then just split the data
        if len(where) == 0:
            yield from super(MinimumNaNSplit, self).split(X, y, groups)
            return
        elif (n_non_nan := not_where.shape[0]) < (n_min := self.min_non_nan +
                                                  1):
            raise ValueError(f"Need at least {n_min} non-nan values, but only"
                             f" have {n_non_nan}")

        check = {'train': lambda t: np.setdiff1d(not_where, t,
                                                 assume_unique=True),
                 'test': lambda t: np.intersect1d(not_where, t,
                                                  assume_unique=True)}

        splits = super().split(X, y, groups)

        # check that all training sets for each kfold within each repetition
        # have at least min_non_nan non-nan values
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
                if check[self.which](test).shape[0] < self.min_non_nan:
                    for _ in range(i + 1, self.n_splits):
                        next(splits)
                    extra = super().split(X, y, groups)
                    one_rep = itertools.islice(extra, self.n_splits)
                    splits = itertools.chain(one_rep, splits)
                    break
                kfold_set[i] = (train, test)
            else:
                yield from kfold_set

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
        cats = np.unique(labels)
        gt_labels = [0] * cats.shape[0]
        min_trials *= self.n_splits
        i = 0
        while not all(g >= min_trials for g in gt_labels):
            np.random.shuffle(labels)
            for j, l in enumerate(cats):
                eval_arr = np.take(arr, np.flatnonzero(labels == l), trials_ax)
                gt_labels[j] = np.min(np.sum(
                    np.all(~np.isnan(eval_arr), axis=2), axis=trials_ax))
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
    """Generate random numbers using sort-based sampling

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
    """
    return np.argsort(np.random.rand(iterations, n_range), axis=1
                      )[:, :n_picks]


def mixup2(arr: np.ndarray, labels: np.ndarray, obs_axis: int,
           alpha: float = 1., seed: int = None) -> None:
    """Mixup the data using the labels

    Parameters
    ----------
    arr : array
        The data to mixup.
    labels : array
        The labels to use for mixing.
    obs_axis : int
        The axis along which to apply func.
    alpha : float
        The alpha value for the beta distribution.
    seed : int
        The seed for the random number generator.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> mixup2(arr, labels, 0)
    >>> arr
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [6.03943491, 7.03943491]])
           """
    if arr.ndim > 2:
        arr = arr.swapaxes(obs_axis, -2)
        for i in range(arr.shape[0]):
            mixup2(arr[i], labels, obs_axis, alpha, seed)
    else:
        if seed is not None:
            np.random.seed(seed)
        if obs_axis == 1:
            arr = arr.T

        is_nan = np.isnan(arr).any(axis=1)
        n_nan = np.where(is_nan)[0]
        n_non_nan = np.where(~is_nan)[0]

        for i in n_nan:
            l_class = labels[i]
            possible_choices = np.nonzero(np.logical_and(
                ~is_nan, labels == l_class))[0]
            choice1 = np.random.choice(possible_choices)
            choice2 = np.random.choice(n_non_nan)
            lam = np.random.beta(alpha, alpha)
            if lam < .5:
                lam = 1 - lam
            arr[i] = lam * arr[choice1] + (1 - lam) * arr[choice2]



def resample(arr: np.ndarray, sfreq: int, new_sfreq: int, axis: int = -1
             ) -> np.ndarray:
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
    >>> np.random.seed(0)
    >>> arr = np.random.rand(10)
    >>> resample(arr, 10, 5)
    array([0.5488135 , 0.60276338, 0.4236548 , 0.43758721, 0.96366276])
    >>> arr = np.arange(10)
    >>> resample(arr, 10, 20)
    array([0.        , 0.47368421, 0.94736842, 1.42105263, 1.89473684,
           2.36842105, 2.84210526, 3.31578947, 3.78947368, 4.26315789,
           4.73684211, 5.21052632, 5.68421053, 6.15789474, 6.63157895,
           7.10526316, 7.57894737, 8.05263158, 8.52631579, 9.        ])
    >>> resample(arr, 10, 7)
    array([0.        , 1.30434783, 2.60869565, 3.91304348, 5.2173913 ,
           6.52173913, 7.82608696])
    >>> arr = np.arange(30).reshape(5, 6)
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
    """
    while axis < 0:
        axis += arr.ndim
    if sfreq == new_sfreq:
        return arr
    elif sfreq > new_sfreq and sfreq % new_sfreq == 0: # simple downsample
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(None, None, sfreq // new_sfreq)
        return arr[tuple(idx)]
    elif sfreq < new_sfreq: # Upsample
        seconds = arr.shape[axis] / sfreq
        o_indices = np.arange(arr.shape[axis])
        new_samps = int(round(new_sfreq * seconds))
        indices = np.linspace(0, arr.shape[axis] - 1, new_samps)
        if arr.ndim == 1:
            return np.interp(indices, o_indices, arr)

        # for multi-dimensional arrays, we flatten non-axis dimensions, then
        # apply the 1d interpolation, then reshape
        func = partial(np.interp, indices, o_indices)
        arr_in = np.swapaxes(arr, axis, -1).reshape(-1, arr.shape[axis])
        out_flat = np.apply_along_axis(func, 1, arr_in)
        out_shape = arr.shape[:axis] + (new_samps,) + arr.shape[axis + 1:]
        return out_flat.reshape(out_shape)
    else:
        # for speed, halve the data until we get close to the desired sfreq
        while sfreq > 2 * new_sfreq and sfreq % 2 == 0:
            arr = resample(arr, sfreq, sfreq // 2, axis)
            sfreq //= 2

        # find freq such that sfreq * freq / new_sfreq is an integer
        freq = int(np.lcm(sfreq, new_sfreq))

        # upsample then downsample
        arr = resample(arr, sfreq, freq, axis)
        return resample(arr, freq, new_sfreq, axis)

