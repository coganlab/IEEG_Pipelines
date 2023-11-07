import numpy as np
from typing import Literal, Tuple
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold
from numba import njit

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
    # >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> oversample_nan(arr, normnd, 0)  # doctest: +ELLIPSIS
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [...
    >>> oversample_nan(arr, mixupnd, 0)  # doctest: +ELLIPSIS
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [...
    >>> arr3 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr3[0, 2, :] = [float("nan")] * 4
    >>> oversample_nan(arr3, mixupnd, 1)  # doctest: +SKIP
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ ...

           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> oversample_nan(arr3, normnd, 1)  # doctest: +SKIP
    array([[[0.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00],
            [4.00000000e+00, 5.00000000e+00, 6.00000000e+00, 7.00000000e+00],
            [...

           [[1.20000000e+01, 1.30000000e+01, 1.40000000e+01, 1.50000000e+01],
            [1.60000000e+01, 1.70000000e+01, 1.80000000e+01, 1.90000000e+01],
            [2.00000000e+01, 2.10000000e+01, 2.20000000e+01, 2.30000000e+01]]])
    # >>> oversample_nan(arr3, normnd, 0)  # doctest: +ELLIPSIS
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
    >>> find_nan_indices(arr, 0)
    (array([3], dtype=int64), array([0, 1, 2], dtype=int64))

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
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> mixupnd(arr, 0)
    >>> arr # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [...
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
    >>> np.random.seed(0)
    >>> arr = np.array([1, 2, 4, 5, 7, 8,
    ... float("nan"), float("nan")])
    >>> normnd(arr)
    >>> arr # doctest: +ELLIPSIS
    array([1.        , 2.        , 4.        , 5.        , 7.        ,
           8.        , ...
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
    .. [1] <https://stackoverflow.com/questions/31955660/efficiently-generating
    -multiple-instances-of-numpy-random-choice-without-replace/31958263#31958263>
    """
    return np.argsort(np.random.rand(iterations, n_range), axis=1
                      )[:, :n_picks]


@njit("void(float64[:])", nogil=True)
def norm1d(arr: Vector) -> None:
    """Oversample by obtaining the distribution and randomly selecting"""
    # Get indices of rows with NaN values
    wh = np.isnan(arr)
    non_nan_rows = np.flatnonzero(~wh)

    # Check if there are at least two non-NaN rows
    if len(non_nan_rows) < 1:
        raise ValueError("No test data to fit distribution")

    # Calculate mean and standard deviation for each column
    mean = np.mean(arr[non_nan_rows])
    std = np.std(arr[non_nan_rows])

    # Get the normal distribution of each timepoint
    for i in np.flatnonzero(wh):
        arr[i] = np.random.normal(mean, std)


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
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> mixup2d(arr)
    >>> arr  #doctest: +ELLIPSIS
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [...
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
