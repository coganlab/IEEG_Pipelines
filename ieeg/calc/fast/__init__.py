import numpy as np
from ieeg.calc.fast.stats import mean_diff as _md
from ieeg.calc.fast.mixup import mixupnd as cmixup, normnd as cnorm
from ieeg.calc.fast.permgt import permgtnd as permgt
from ieeg.calc.fast.concat import nan_concatinate

__all__ = ["mean_diff", "mixup", "permgt", "norm", "concatenate_arrays"]


def concatenate_arrays(arrays: tuple[np.ndarray, ...], axis: int = 0
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

    Examples
    --------
    >>> concatenate_arrays((np.array([1, 2, 3]), np.array([4, 5])), axis=None)
    array([[ 1.,  2.,  3.],
           [ 4.,  5., nan]])
    >>> concatenate_arrays((np.array([1, 2, 3]), np.array([4, 5])), axis=0)
    array([1., 2., 3., 4., 5.])
    >>> arr1 = np.arange(6, dtype=float).reshape(1, 2, 3)
    >>> arr2 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> concatenate_arrays((arr1[0], arr2[0]), axis=0)
    array([[ 0.,  1.,  2., nan],
           [ 3.,  4.,  5., nan],
           [ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])
    >>> concatenate_arrays((arr2[0], arr1[0]), axis=1)
    array([[ 0.,  1.,  2.,  3.,  0.,  1.,  2.],
           [ 4.,  5.,  6.,  7.,  3.,  4.,  5.],
           [ 8.,  9., 10., 11., nan, nan, nan]])
    >>> arr = concatenate_arrays((arr1[0], arr2[0]), axis=None)
    >>> arr
    array([[[ 0.,  1.,  2., nan],
            [ 3.,  4.,  5., nan],
            [nan, nan, nan, nan]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]])
    """

    if axis is None:
        axis = 0
        arrays = [np.expand_dims(ar, axis) for ar in arrays]

    arrays = [ar.astype(float) for ar in arrays if ar.size > 0]

    while axis < 0:
        axis += max(a.ndim for a in arrays)

    return nan_concatinate(arrays, axis)


def mixup(arr: np.ndarray, obs_axis: int, alpha: float = 1.,
          seed: int = -1) -> None:
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
    >>> arr = np.array([[1, 2], [4, 5], [7, 8],
    ... [float("nan"), float("nan")]])
    >>> mixup(arr, 0, seed=42)
    >>> arr # doctest: +NORMALIZE_WHITESPACE
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.24946679, 6.24946679]])
    >>> arr2 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr2[0, 2, :] = [float("nan")] * 4
    >>> mixup(arr2, 1, seed=42)
    >>> arr2 # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 2.33404428,  3.33404428,  4.33404428,  5.33404428]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> arr3 = np.arange(24, dtype=float).reshape(3, 2, 4)
    >>> arr3[0, :, :] = float("nan")
    >>> mixup(arr3, 0, seed=42)
    >>> arr3 # doctest: +NORMALIZE_WHITESPACE
    array([[[12.66808855, 13.66808855, 14.66808855, 15.66808855],
            [17.31717879, 18.31717879, 19.31717879, 20.31717879]],
    <BLANKLINE>
           [[ 8.        ,  9.        , 10.        , 11.        ],
            [12.        , 13.        , 14.        , 15.        ]],
    <BLANKLINE>
           [[16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    """
    cmixup(arr, obs_axis, alpha, seed)


def norm(arr: np.ndarray, obs_axis: int = -1) -> None:
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
    >>> norm(arr)
    >>> arr
    array([1.        , 2.        , 4.        , 5.        , 7.        ,
           8.        , 8.91013086, 5.50039302])
    """
    cnorm(arr, obs_axis)


def mean_diff(group1: np.ndarray, group2: np.ndarray,
              axis: int = -1) -> np.ndarray | float:
    """Calculate the mean difference between two groups.

    This function is the default statistic function for time_perm_cluster. It
    calculates the mean difference between two groups along the specified axis.

    Parameters
    ----------
    group1 : array, shape (..., time)
        The first group of observations.
    group2 : array, shape (..., time)
        The second group of observations.
    axis : int or tuple of ints, optional
        The axis or axes along which to compute the mean difference. If None,
        compute the mean difference over all axes.

    Returns
    -------
    avg1 - avg2 : array or float
        The mean difference between the two groups.

    Examples
    --------
    >>> import numpy as np
    >>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]], order='F').T
    >>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], order='F').T
    >>> mean_diff(group1, group2, axis=0)
    array([ 0., 14.])
    >>> mean_diff(group1, group2, axis=1)
    array([ 0., 30.,  0.,  5.,  0.])
    """
    in1 = np.moveaxis(group1, axis, -1)
    in2 = np.moveaxis(group2, axis, -1)

    return _md(in1, in2)
