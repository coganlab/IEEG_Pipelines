import numpy as np
from typing import Literal, Tuple
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold
from numba import njit

Array2D = NDArray[Tuple[Literal[2], ...]]
Vector = NDArray[Literal[1]]


class TwoSplitNaN(RepeatedStratifiedKFold):
    """A Repeated Stratified KFold iterator that splits the data into sections
    that do and don't contain NaNs"""

    def __init__(self, n_splits, n_repeats=10, random_state=None):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats,
                         random_state=random_state)
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):

        # find where the nans are
        where = np.isnan(X).any(axis=tuple(range(X.ndim))[1:])
        not_where = np.where(~where)[0]
        where = np.where(where)[0]

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

            train = np.concatenate((where[nan_train], not_where[not_nan_train]))
            test = np.concatenate((where[nan_test], not_where[not_nan_test]))
            yield train, test


def oversample_nan(arr: np.ndarray, func: callable, copy: bool = True) -> np.ndarray:
    """Oversample nan rows using func

    Parameters
    ----------
    arr : array
        The data to oversample.
    func : callable
        The function to use to oversample the data.
    copy : bool
        Whether to copy the data before oversampling.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> oversample_nan(arr, smote)
    array([[1.       , 2.       ],
           [4.       , 5.       ],
           [7.       , 8.       ],
           [2.1455681, 2.1455681]])
    >>> arr3 = np.stack([arr] * 3)
    >>> arr3[0, 2, :] = [float("nan")] * 2
    >>> oversample_nan(arr3, mixup)[0]
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [2.31225044, 3.31225044],
           [2.31225044, 3.31225044]])
    """

    if copy:
        arr = arr.copy()

    if arr.ndim == 1:
        raise ValueError("Cannot apply SMOTE to a 1-dimensional array")
    elif arr.ndim > 2:
        for i in range(arr.shape[0]):
            oversample_nan(arr[i], func, copy=False)
    else:
        func(arr)

    return arr


@njit(cache=True)
def find_nan_indices(arr: Array2D) -> tuple:
    """Find the indices of rows with and without NaN values

    Parameters
    ----------
    arr : array
        The data to find indices.

    Returns
    -------
    tuple
        A tuple of two arrays containing the indices of rows with and without NaN values.

    Examples
    --------
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> find_nan_indices(arr)
    (array([3], dtype=int64), array([0, 1, 2], dtype=int64))

    """

    # Initialize boolean mask of rows with NaN values
    nan = np.zeros(arr.shape[0], dtype=np.bool_)

    # Check each row individually
    for i in range(arr.shape[0]):
        nan[i] = np.any(np.isnan(arr[i]))

    # Get indices of rows with and without NaN values using boolean indexing
    nan_rows = np.nonzero(nan)[0]
    non_nan_rows = np.nonzero(~nan)[0]

    return nan_rows, non_nan_rows


@njit(cache=True)
def smote(arr: Array2D) -> None:
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


@njit(cache=True)
def norm(arr: Array2D) -> None:
    """Oversample by obtaining the distribution and randomly selecting"""
    # Get indices of rows with NaN values
    nan_rows, non_nan_rows = find_nan_indices(arr)

    # Check if there are at least two non-NaN rows
    if len(non_nan_rows) < 1:
        raise ValueError("No test data to fit distribution")

    # Initialize mean and standard deviation arrays
    mean = np.zeros(arr.shape[1])
    std = np.zeros(arr.shape[1])

    # Calculate mean and standard deviation for each column
    for i in range(arr.shape[1]):
        mean[i] = np.mean(arr[non_nan_rows, i])
        std[i] = np.std(arr[non_nan_rows, i])

    # Get the normal distribution of each timepoint
    for row in nan_rows:
        for j in range(arr.shape[1]):
            arr[row, j] = np.random.normal(mean[j], std[j])


@njit(cache=True)
def mixup(arr: Array2D, alpha: float = 1.) -> None:
    # Get indices of rows with NaN values
    nan_rows, non_nan_rows = find_nan_indices(arr)
    n_nan = len(nan_rows)

    # Check if there are at least two non-NaN rows
    if len(non_nan_rows) < 2:
        raise ValueError("Not enough non-NaN rows to apply mixup algorithm")

    # Construct an array of 2-length vectors for each NaN row
    vectors = np.empty((n_nan, 2))

    # The two elements of each vector are different indices of non-NaN rows
    for i in range(n_nan):
        vectors[i, :] = np.random.choice(non_nan_rows, 2, replace=False)

    # get beta distribution parameters
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x1 = arr[vectors[:, 0].astype(np.intp)]
    x2 = arr[vectors[:, 1].astype(np.intp)]

    arr[nan_rows] = lam * x1 + (1 - lam) * x2
