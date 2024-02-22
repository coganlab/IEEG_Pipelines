import itertools
from typing import Literal, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold

import itertools
from ieeg.calc.mixup import mixupnd as cmixup

Array2D = NDArray[Tuple[Literal[2], ...]]
Vector = NDArray[Literal[1]]


class TwoSplitNaN(RepeatedStratifiedKFold):
    """A Repeated Stratified KFold iterator that splits the data into sections
    that do and don't contain NaNs

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
    >>> from ieeg.calc.oversample import TwoSplitNaN
    >>> np.random.seed(0)
    >>> X = np.vstack((np.arange(1, 9).reshape(4, 2), np.full((4, 2), np.nan)))
    >>> y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> tsn = TwoSplitNaN(2, 3)
    >>> for train, test in tsn.split(X, y):
    ...     print("train:", train, "test:", test)
    train: [1 2 4 7] test: [0 3 5 6]
    train: [0 3 5 6] test: [1 2 4 7]
    train: [1 3 5 7] test: [0 2 4 6]
    train: [0 2 4 6] test: [1 3 5 7]
    train: [1 2 5 7] test: [0 3 4 6]
    train: [0 3 4 6] test: [1 2 5 7]
    """

    def __init__(self, n_splits: int, n_repeats: int = 10,
                 random_state: int = None):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats,
                         random_state=random_state)
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):

        # find where the nans are
        where = np.isnan(X).any(axis=tuple(range(X.ndim))[1:])
        not_where = np.where(~where)[0]
        where = np.where(where)[0]

        # if there are no nans, then just split the data
        if len(where) == 0:
            yield from super(TwoSplitNaN, self).split(X, y, groups)
            return

        # split the data
        nan = X[where, ...]
        not_nan = X[not_where, ...]

        # split the labels and verify the stratification
        y_nan = y[where, ...]
        y_not_nan = y[not_where, ...]
        for i in set(y_nan):
            least = np.sum(y_not_nan == i)
            if np.sum(y_not_nan == i) < self.n_splits:
                raise ValueError(f"Cannot split data into {self.n_splits} "
                                 f"folds with at most {least} non nan values")

        # split each section into k folds
        nan_folds = super().split(nan, y_nan)
        not_nan_folds = super().split(not_nan, y_not_nan)

        # combine the folds
        for (nan_train, nan_test), (not_nan_train, not_nan_test) in zip(
                nan_folds, not_nan_folds):
            train = np.concatenate((where[nan_train], not_where[not_nan_train]
                                    ))
            test = np.concatenate((where[nan_test], not_where[not_nan_test]))
            train.sort()
            test.sort()
            yield train, test


@njit(nogil=True, cache=True)
def mixupnd(arr: np.ndarray, obs_axis: int, alpha: float = 1.) -> None:
    """Oversample by mixing two random non-NaN observations

    Parameters
    ----------
    arr : array
        The data to oversample.
    obs_axis : int
        The axis along which to apply func.
    alpha : float
        The alpha parameter for the beta distribution. If alpha is 0, then
        the distribution is uniform. If alpha is 1, then the distribution is
        symmetric. If alpha is greater than 1, then the distribution is
        skewed towards the first observation. If alpha is less than 1, then
        the distribution is skewed towards the second observation.

    Examples
    --------
    >>> from ieeg.calc.oversample import mixupnd
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> mixupnd(arr, 0)
    >>> arr # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[...
    """

    # create a view of the array with the observation axis in the second to
    # last position
    arr_in = np.swapaxes(arr, obs_axis, -2)

    if arr.ndim == 2:
        mixup2d(arr_in, alpha)
    elif arr.ndim > 2:
        for ijk in np.ndindex(arr.shape[:-2]):
            mixup2d(arr_in[ijk], alpha)
    else:
        raise ValueError("Cannot apply mixup to a 1-dimensional array")


@njit(["void(f8[:, :], Omitted(1.))", "void(f8[:, :], f8)"], nogil=True)
def mixup2d(arr: Array2D, alpha: float = 1.) -> None:
    """Oversample by mixing two random non-NaN observations

    Parameters
    ----------
    arr : array
        The data to oversample.
    alpha : float
        The alpha parameter for the beta distribution. If alpha is 0, then
        the distribution is uniform. If alpha is 1, then the distribution is
        symmetric. If alpha is greater than 1, then the distribution is
        skewed towards the first observation. If alpha is less than 1, then
        the distribution is skewed towards the second observation.

    Examples
    --------
    >>> from ieeg import _rand_seed
    >>> _rand_seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> mixup2d(arr)
    >>> arr
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.72901614, 6.72901614]])
    """
    # Get indices of rows with NaN values
    wh = np.zeros(arr.shape[0], dtype=np.bool_)
    for i in range(arr.shape[0]):
        wh[i] = np.any(np.isnan(arr[i]))
    non_nan_rows = np.flatnonzero(~wh)
    n_nan = np.sum(wh)

    # Construct an array of 2-length vectors for each NaN row
    vectors = np.empty((n_nan, 2))

    # The two elements of each vector are different indices of non-NaN rows
    for i in range(n_nan):
        vectors[i, :] = np.random.choice(non_nan_rows, 2, replace=False)

    # get beta distribution parameters
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x1 = arr[vectors[:, 0].astype(np.intp)]
    x2 = arr[vectors[:, 1].astype(np.intp)]

    arr[wh] = lam * x1 + (1 - lam) * x2


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
    >>> from ieeg.calc.oversample import TwoSplitNaN
    >>> from ieeg import _rand_seed
    >>> _rand_seed(0)
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
    """

    def __init__(self, n_splits: int, n_repeats: int = 10,
                 random_state: int = None, min_non_nan: int = 2):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats,
                         random_state=random_state)
        self.n_splits = n_splits
        self.min_non_nan = min_non_nan

    def split(self, X, y=None, groups=None):

        # find where the nans are
        where = np.isnan(X).any(axis=tuple(range(X.ndim))[1:])
        not_where = np.where(~where)[0]
        where = np.where(where)[0]

        # if there are no nans, then just split the data
        if len(where) == 0:
            yield from super(MinimumNaNSplit, self).split(X, y, groups)
            return
        elif (n_non_nan := len(not_where)) < (n_min := self.min_non_nan + 1):
            raise ValueError(f"Need at least {n_min} non-nan values, but only"
                             f" have {n_non_nan}")

        splits = super().split(X, y, groups)

        # check that all training sets for each kfold within each repetition
        # have at least min_non_nan non-nan values
        while element := next(splits, False):
            kfold_set = []
            for i in range(self.n_splits):
                if i == 0:
                    train, test = element
                else:
                    train, test = next(splits)

                # if any test set has more non-nan values than the total number
                # of non-nan values minus the minimum number of non-nan values,
                # then throw out the split and append an extra repetition
                if sum(ix not in test for ix in not_where) < self.min_non_nan:
                    for _ in range(i + 1, self.n_splits):
                        next(splits)
                    extra = super().split(X, y, groups)
                    one_rep = itertools.islice(extra, self.n_splits)
                    splits = itertools.chain(splits, one_rep)
                    break
                kfold_set.append((train, test))
            else:
                yield from kfold_set

    @staticmethod
    def oversample(arr: np.ndarray, func: callable = cmixup,
                   axis: int = 1, copy: bool = True) -> np.ndarray:
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
        >>> from ieeg import _rand_seed
        >>> _rand_seed(0)
        >>> np.random.seed(0)
        >>> arr = np.array([[1, 2], [4, 5], [7, 8],
        ... [float("nan"), float("nan")]])
        >>> MinimumNaNSplit.oversample(arr, normnd, 0)
        array([[1.        , 2.        ],
               [4.        , 5.        ],
               [7.        , 8.        ],
               [8.32102813, 5.98018098]])
        >>> MinimumNaNSplit.oversample(arr, mixupnd, 0)
        array([[1.        , 2.        ],
               [4.        , 5.        ],
               [7.        , 8.        ],
               [3.13990284, 4.13990284]])
        """
        if copy:
            arr = arr.copy()

        axis = arr.ndim + axis if axis < 0 else axis

        if arr.ndim <= 0:
            raise ValueError("Cannot apply func to a 0-dimensional array")
        else:
            func(arr, axis)
        return arr

    def shuffle_labels(self, arr: np.ndarray, labels: np.ndarray,
                       trials_ax: int = 1, min_trials: int = None):
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
        >>> MinimumNaNSplit(1).shuffle_labels(arr, labels)
        >>> labels
        array([1, 1, 0, 0])
        """
        cats = np.unique(labels)
        gt_labels = [0] * cats.shape[0]
        if min_trials is None:
            min_trials = self.n_splits
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
                   copy: bool = True) -> np.ndarray:
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
    >>> from ieeg import _rand_seed
    >>> _rand_seed(0)
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> oversample_nan(arr, normnd, 0)
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [8.32102813, 5.98018098]])
    >>> oversample_nan(arr, mixupnd, 0)
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [3.13990284, 4.13990284]])
    >>> arr3 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr3[0, 2, :] = [float("nan")] * 4
    >>> oversample_nan(arr3, mixupnd, 1)
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 0.61989218,  1.61989218,  2.61989218,  3.61989218]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> oversample_nan(arr3, normnd, 1)
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [-2.85190914,  2.0938884 ,  3.05845799,  6.94603199]],
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


def norm(arr: np.ndarray, obs_axis: int) -> None:
    """A jit-less version of normnd"""
    # Get indices of rows with NaN values
    nan_rows, non_nan_rows = find_nan_indices(arr, obs_axis)

    # Check if there are at least two non-NaN rows
    if len(non_nan_rows) < 1:
        raise ValueError("No test data to fit distribution")

    nan_idx = (non_idx := [slice(None)] * arr.ndim).copy()
    non_idx[obs_axis] = non_nan_rows
    nan_idx[obs_axis] = nan_rows

    # Calculate mean and standard deviation for each column
    mean = np.mean(arr[tuple(non_idx)], axis=obs_axis, keepdims=True)
    std = np.std(arr[tuple(non_idx)], axis=obs_axis, keepdims=True)

    # Get the normal distribution of each timepoint
    out_shape = tuple(arr.shape[i] if i != obs_axis else len(nan_rows)
                      for i in range(arr.ndim))
    arr[tuple(nan_idx)] = np.random.normal(mean, std, out_shape)


@njit(nogil=True, cache=True)
def normnd(arr: np.ndarray, obs_axis: int = -1) -> None:
    """Oversample by obtaining the distribution and randomly selecting

    Parameters
    ----------
    arr : array
        The data to oversample.
    obs_axis : int
        The axis along which to apply func.

    Examples
    --------
    >>> from ieeg import _rand_seed
    >>> _rand_seed(0)
    >>> np.random.seed(0)
    >>> arr = np.array([1, 2, 4, 5, 7, 8,
    ... float("nan"), float("nan")])
    >>> normnd(arr)
    >>> arr
    array([1.        , 2.        , 4.        , 5.        , 7.        ,
           8.        , 8.91013086, 5.50039302])
    """

    # create a view of the array with the observation axis in the last position
    arr_in = np.swapaxes(arr, obs_axis, -1)
    if arr.ndim == 1:
        norm1d(arr_in)
    elif arr.ndim > 1:
        for ijk in np.ndindex(arr_in.shape[:-1]):
            norm1d(arr_in[ijk])
    else:
        raise ValueError("Cannot apply norm to a 0-dimensional array")


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


# @njit("void(float64[:])", nogil=True)
# def norm1d(arr: Vector) -> None:
#     """Oversample by obtaining the distribution and randomly selecting"""
#     # Get indices of rows with NaN values
#     wh = np.isnan(arr)
#     non_nan_rows = np.flatnonzero(~wh)
#
#     # Check if there are at least two non-NaN rows
#     if len(non_nan_rows) < 1:
#         raise ValueError("No test data to fit distribution")
#
#     # Calculate mean and standard deviation for each column
#     mean = np.mean(arr[non_nan_rows])
#     std = np.std(arr[non_nan_rows])
#
#     # Get the normal distribution of each timepoint
#     for i in np.flatnonzero(wh):
#         arr[i] = np.random.normal(mean, std)


def smote(arr: np.ndarray) -> None:
    """Oversample by mixing two random non-NaN observations

    Parameters
    ----------
    arr : array
        The data to oversample.

    Notes
    -----
        This func assumes that the observations are the second to last axis.
    """
    if arr.ndim < 2:
        raise ValueError("Cannot apply SMOTE to a 1 or 0-dimensional array")
    elif arr.ndim > 2:
        for i in range(arr.shape[0]):
            smote(arr[i])
        return
    # Get indices of rows with NaN values
    nan_rows, non_nan_rows = find_nan_indices(arr)
    n_nan = len(nan_rows)

    # Check if there are at least two non-NaN rows
    if len(non_nan_rows) < 2:
        raise ValueError("Not enough non-NaN rows to apply SMOTE algorithm")

    # Construct an array of 3-length vectors for each NaN row
    vectors = np.empty((n_nan, 3))

    # First two elements of each vector are different indices of non-NaN rows
    for i in range(n_nan):
        vectors[:, :2] = np.random.choice(non_nan_rows, 2, replace=False)

    # The last element of each vector is a random float between 0 and 1
    vectors[:, 2] = np.random.random(n_nan)

    # Compute the differences between the selected non-NaN rows
    diffs = (arr[vectors[:, 0].astype(np.intp)] -
             arr[vectors[:, 1].astype(np.intp)])

    # Multiply the differences by the random multipliers
    arr[nan_rows] = diffs * vectors[:, 2, None]
