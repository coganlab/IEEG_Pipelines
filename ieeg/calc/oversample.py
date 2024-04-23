from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold

import itertools
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
