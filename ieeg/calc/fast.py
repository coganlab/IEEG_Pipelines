import numpy as np
from ieeg.calc._fast.ufuncs import mean_diff as _md, t_test as _ttest
from ieeg.calc._fast.mixup import mixupnd as cmixup, normnd as cnorm
from ieeg.calc._fast.permgt import permgtnd as permgt
from ieeg.arrays.api import array_namespace, is_numpy, is_torch, Array, is_cupy
from scipy.stats import rankdata
from functools import partial

__all__ = ["mean_diff", "mixup", "permgt", "norm", "concatenate_arrays",
           "ttest", "brunnermunzel"]


def brunnermunzel(x: np.ndarray, y: np.ndarray, axis=None, nan_policy='omit'):
    """
    Compute the Brunner-Munzel test statistic for two independent samples.

    The Brunner-Munzel test is used to compare the stochastic dominance of two
    independent samples and does not assume equal variances. It is a
     nonparametric statistical test that operates using ranked data. This
      implementation allows handling NaN values based on the specified policy.

    Parameters
    ----------
    x : np.ndarray
        The first input array representing sample data.
    y : np.ndarray
        The second input array representing sample data.
    axis : int or None, optional
        The axis along which to compute the test statistic. If None, the arrays
        are flattened before computation. Default is None.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle NaN values in the inputs:
        - 'propagate': Returns NaN in the result if NaN is present in the input
        - 'raise': Raises an error if NaN is detected in the input.
        - 'omit': Omits NaN values during the computation.
        Default is 'omit'.

    Returns
    -------
    np.ndarray or float
        The computed Brunner-Munzel statistic, returned as a scalar if the
         input arrays are 1D and as an array otherwise.
    """

    if axis is None:
        nx, ny = x.size, y.size
        idxx = slice(0, nx)
        idxy = slice(nx, nx+ny)
        x, y = x.flat, y.flat
        concat = np.concatenate((x, y), axis=0)
    else:
        while axis < 0:
            axis += x.ndim
        nx, ny = x.shape[axis], y.shape[axis]
        idxx = tuple(slice(None) if i != axis else slice(0, nx)
                     for i in range(x.ndim))
        idxy = tuple(slice(None) if i != axis else slice(nx, nx+ny)
                     for i in range(x.ndim))
        concat = np.concatenate((x, y), axis=axis)

    where = ~np.isnan(concat)
    if nan_policy == 'omit':
        rank = partial(rankdata, nan_policy=nan_policy)
        wherex, wherey = where[idxx], where[idxy]
    else:
        rank = rankdata
        wherex = wherey = None
        if np.any(~where) and nan_policy == 'raise':
            raise ValueError("The input contains NaN.")

    kwargsx = dict(axis=axis, where=wherex, keepdims=True)
    kwargsy = dict(axis=axis, where=wherey, keepdims=True)

    rankc = rank(concat, axis=axis)
    rankcx, rankcy = rankc[idxx], rankc[idxy]
    rankcx_mean, rankcy_mean = rankcx.mean(**kwargsx), rankcy.mean(**kwargsy)
    rankx, ranky = rank(x, axis=axis), rank(y, axis=axis)
    rankx_mean, ranky_mean = rankx.mean(**kwargsx), ranky.mean(**kwargsy)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0),
                **kwargsx) / (nx - 1)
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0),
                **kwargsy) / (ny - 1)

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
    return np.squeeze(wbfn)


def ttest(group1: np.ndarray, group2: np.ndarray,
          axis: int, xp=None) -> np.ndarray:
    """Calculate the t-statistic between two groups.

    This function is the default statistic function for time_perm_cluster. It
    calculates the t-statistic between two groups along the specified axis.

    Parameters
    ----------
    group1 : array, shape (..., time)
        The first group of observations.
    group2 : array, shape (..., time)
        The second group of observations.
    axis : int or tuple of ints, optional
        The axis or axes along which to compute the t-statistic. If None,
        compute the t-statistic over all axes.

    Returns
    -------
    t : array
        The t-statistic between the two groups.

    Examples
    --------
    >>> import numpy as np
    >>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]])
    >>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    >>> ttest(group1, group2, 1)
    array([      nan, 1.2004901])
    >>> ttest(group1, group2, 0)
    array([0.        , 1.01680311, 0.        , 1.10431526, 0.        ])
    >>> import cupy as cp # doctest: +SKIP
    >>> group1 = cp.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]]
    ... ) # doctest: +SKIP
    >>> group2 = cp.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]) # doctest: +SKIP
    >>> ttest(group1, group2, 1) # doctest: +SKIP
    array([      nan, 1.2004901])
    """
    if xp is None:
        xp = array_namespace(group1, group2)
    if is_numpy(xp):
        return _ttest(group1, group2, axes=[axis, axis])
    elif is_cupy(xp):
        n1 = xp.sum(~xp.isnan(group1), axis=axis)
        n2 = xp.sum(~xp.isnan(group2), axis=axis)
        mean1 = xp.nansum(group1, axis=axis) / n1
        mean2 = xp.nansum(group2, axis=axis) / n2
        var1 = xp.nanvar(group1, axis=axis)
        var2 = xp.nanvar(group2, axis=axis)
        return (mean1 - mean2) / xp.sqrt(var1 / (n1 - 1) + var2 / (n2 - 1))
    else:
        raise NotImplementedError("T-test is not implemented for this array"
                                  " type.")


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
    >>> concatenate_arrays((np.array([[1, 2, 3]]), np.array([[4, 5]])), axis=0)
    array([[ 1.,  2.,  3.],
           [ 4.,  5., nan]])
    >>> concatenate_arrays((np.array([1, 2, 3]), np.array([4, 5])), axis=0)
    array([1., 2., 3., 4., 5.])
    >>> arr1 = np.arange(60, dtype=float).reshape(10, 2, 3)
    >>> arr2 = np.arange(240, dtype=float).reshape(20, 3, 4)
    >>> concatenate_arrays((arr1, arr2), axis=0)[0]
    array([[ 0.,  1.,  2., nan],
           [ 3.,  4.,  5., nan],
           [nan, nan, nan, nan]])
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
    >>> concatenate_arrays((arr2[0].astype('f2'), arr1[0].astype('f2')),
    ... axis=1)
    array([[ 0.,  1.,  2.,  3.,  0.,  1.,  2.],
           [ 4.,  5.,  6.,  7.,  3.,  4.,  5.],
           [ 8.,  9., 10., 11., nan, nan, nan]], dtype=float16)
    """

    if axis is None:
        axis = 0
        arrays = [np.expand_dims(ar, axis) for ar in arrays]

    arrays = [ar.astype(float) if ar.dtype.kind in 'iu' else ar
              for ar in arrays if ar.size > 0]
    if len(arrays) == 0:
        return np.array([])

    while axis < 0:
        axis += max(a.ndim for a in arrays)

    max_shape = [max(a.shape[ax] for a in arrays) if ax != axis else
                 sum(a.shape[ax] for a in arrays)
                 for ax in range(arrays[0].ndim)]
    out = np.full(max_shape, np.nan, dtype=arrays[0].dtype)
    start = 0
    for i, ar in enumerate(arrays):
        slices = tuple(slice(start, start + ar.shape[ax]) if ax == axis else
                       slice(ar.shape[ax]) for ax in range(ar.ndim))
        out[slices] = ar
        start += ar.shape[axis]
    return out


def _mixup_np(arr: np.ndarray, obs_axis: int, alpha: float = 1.,
              rng: int = None) -> None:
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
    >>> _mixup_np(arr, 0, rng=42)
    >>> arr # doctest: +NORMALIZE_WHITESPACE +SKIP
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.24946679, 6.24946679]])
    >>> arr2 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr2[0, 2, :] = [float("nan")] * 4
    >>> _mixup_np(arr2, 1, rng=42)
    >>> arr2 # doctest: +NORMALIZE_WHITESPACE +SKIP
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 2.33404428,  3.33404428,  4.33404428,  5.33404428]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> arr3 = np.arange(24).reshape(3, 2, 4).astype("f2")
    >>> arr3[0, :, :] = float("nan")
    >>> _mixup_np(arr3, 0, rng=42)
    >>> arr3 # doctest: +NORMALIZE_WHITESPACE +SKIP
    array([[[12.67, 13.67, 14.67, 15.67],
            [17.31, 18.31, 19.31, 20.31]],
    <BLANKLINE>
           [[ 8.  ,  9.  , 10.  , 11.  ],
            [12.  , 13.  , 14.  , 15.  ]],
    <BLANKLINE>
           [[16.  , 17.  , 18.  , 19.  ],
            [20.  , 21.  , 22.  , 23.  ]]], dtype=float16)
    """

    if obs_axis == 0:
        arr = arr.swapaxes(1, obs_axis)
    if arr.ndim > 3:
        for i in range(arr.shape[0]):
            _mixup_np(arr[i], obs_axis - 1, alpha, rng)
    elif arr.ndim == 1:
        raise ValueError("Array must have at least 2 dimensions")
    else:
        if rng is None:
            rng = np.random.randint(0, 2 ** 16 - 1)

        if arr.dtype != np.float64:
            temp = arr.astype('f8', copy=True)
            cmixup(temp, 1, alpha, rng)
            arr[...] = temp
        else:
            cmixup(arr, 1, alpha, rng)


def mixup(arr: Array, obs_axis: int, alpha: float = 1.,
          rng=None) -> None:
    """Replace rows along the observation axis that are “missing” (i.e. contain
     any NaNs) with a random convex combination of two non‐missing rows (the
      “mixup”).

    This function works for arrays of arbitrary dimension so long as the
    observation axis (obs_axis) contains the “rows” to mix up and the last axis
    holds features. In higher dimensions the axes other than obs_axis and the
    last axis are treated as independent batch indices. (Every such batch is
     assumed to have at least one non-NaN row.)

    The mixup coefficient for each missing row is drawn from a beta
     distribution with parameters (alpha, alpha) and then “flipped” if it is
      less than 0.5 (so that the coefficient is always >=0.5).

    Parameters
    ----------
    arr : np.ndarray
        Array of data. In the 2D case it should have shape (n_obs, n_features).
        For higher dimensions, the last axis is taken as features and obs_axis
        (which must not be the last axis) is the observation axis.
    obs_axis : int
        The axis along which to look for rows that contain any NaN.
    alpha : float, default=1.
        The alpha parameter for the beta distribution.
    rng : np.random.RandomState or similar, optional
        A random number generator (if None, one is created using
         np.random.RandomState()).

    Returns
    -------
    None; arr is modified in-place.

    Examples
    --------
    >>> arr = np.array([[1, 2],
    ...                 [4, 5],
    ...                 [7, 8],
    ...                 [float("nan"), float("nan")]])
    >>> mixup(arr, 0, rng=42)
    >>> arr # doctest: +SKIP
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [5.24946679, 6.24946679]])

    For a 3D example (here we mix along axis 1):
    >>> arr3 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr3[0, 2, :] = [float("nan")] * 4
    >>> mixup(arr3, 1, rng=42)
    >>> arr3 # doctest: +SKIP
    array([[[ 0.        ,  1.        ,  2.        ,  3.        ],
            [ 4.        ,  5.        ,  6.        ,  7.        ],
            [ 2.33404428,  3.33404428,  4.33404428,  5.33404428]],
    <BLANKLINE>
           [[12.        , 13.        , 14.        , 15.        ],
            [16.        , 17.        , 18.        , 19.        ],
            [20.        , 21.        , 22.        , 23.        ]]])
    >>> np.random.seed(0)
    >>> group2 = np.random.rand(500, 10, 10, 100).astype("float16")
    >>> group2[::2, 0, 0, :] = np.nan
    >>> mixup(group2, 0)
    >>> group2[:10, 0, 0, :5] # doctest: +SKIP
    array([[0.3274 , 0.2805 , 0.1257 , 0.1256 , 0.3027 ],
           [0.748  , 0.1802 , 0.389  , 0.0376 , 0.01179],
           [0.6484 , 0.829  , 0.8213 , 0.2578 , 0.5327 ],
           [0.7583 , 0.5034 , 0.177  , 0.8325 , 0.5166 ],
           [0.7397 , 0.857  , 0.449  , 0.5913 , 0.714  ],
           [0.3076 , 0.062  , 0.989  , 0.719  , 0.758  ],
           [0.571  , 0.176  , 0.679  , 0.6924 , 0.636  ],
           [0.6323 , 0.07513, 0.722  , 0.4668 , 0.7417 ],
           [0.6987 , 0.3787 , 0.4668 , 0.04987, 0.915  ],
           [0.1912 , 0.05853, 0.4368 , 0.72   , 0.824  ]], dtype=float16)
    >>> import cupy as cp # doctest: +SKIP
    >>> group3 = cp.random.randn(100, 10, 10, 100) # doctest: +SKIP
    >>> group3[0::2, 0, 0, :] = float("nan") # doctest: +SKIP
    >>> mixup(group3, 0) # doctest: +SKIP
    >>> group3[0, 0, :, :5] # doctest: +SKIP
    array([[0.3274 , 0.2805 , 0.1257 , 0.1256 , 0.3027 ],
           [0.748  , 0.1802 , 0.389  , 0.0376 , 0.01179],
           [0.6484 , 0.829  , 0.8213 , 0.2578 , 0.5327 ],
           [0.7583 , 0.5034 , 0.177  , 0.8325 , 0.5166 ],
           [0.7397 , 0.857  , 0.449  , 0.5913 , 0.714  ],
           [0.3076 , 0.062  , 0.989  , 0.719  , 0.758  ],
           [0.571  , 0.176  , 0.679  , 0.6924 , 0.636  ],
           [0.6323 , 0.07513, 0.722  , 0.4668 , 0.7417 ],
           [0.6987 , 0.3787 , 0.4668 , 0.04987, 0.915  ],
           [0.1912 , 0.05853, 0.4368 , 0.72   , 0.824  ]], dtype=float16)
    """
    xp = array_namespace(arr)
    if is_numpy(xp):
        _mixup_np(arr, obs_axis, alpha, rng)
        return
    elif is_torch(xp):  # TODO: remove this crutch to keep data on the GPU
        temp = arr.numpy(force=True).astype(float)
        _mixup_np(temp, obs_axis, alpha, rng)
        arr.copy_(xp.from_numpy(temp))
        return

    if rng is None:
        if is_torch(xp):
            xp.random.manual_seed(xp.random.seed())
            rng = xp
            xp.beta = xp.distributions.beta.Beta(alpha, alpha)
        else:
            rng = xp.random.RandomState()

    # Bring the observation axis to the front; this is a view.
    arr_view = xp.moveaxis(arr, obs_axis, 0)

    # For ndim >= 3, assume that the last axis holds features.
    # Flatten all intermediate (batch) dimensions into one.
    n_obs = arr_view.shape[0]
    n_features = arr_view.shape[-1]
    # if is_torch(xp) and not arr_view.is_contiguous():
    #     arr_view = arr_view.contiguous()

    arr_flat = arr_view.reshape(n_obs, -1, n_features)
    # Compute a mask over the observation axis for each batch:
    mask = xp.isnan(arr_view).any(axis=-1).reshape(n_obs, -1)
    # For each batch (i.e. each column in the flattened batch dimension) we
    # want to know the available (non-NaN) indices. We do this by sorting the
    # boolean mask along axis 0: since False sorts before True, the first few
    # indices are the non-missing ones.
    order = xp.argsort(mask, axis=0)
    counts = xp.sum(~mask, axis=0)  # number of non-missing rows per batch
    # Get all indices where the observation is missing.
    missing_rows, batch_idx = xp.nonzero(mask)
    if missing_rows.size:
        L = missing_rows.shape[0]
        # For each missing observation, generate a random index into the
        # available (non-missing) rows in its batch.
        idx1 = xp.astype(rng.rand(L) * counts[batch_idx], int)
        idx2 = xp.astype(rng.rand(L) * counts[batch_idx], int)
        donor1 = order[idx1, batch_idx]
        donor2 = order[idx2, batch_idx]
        if is_torch(xp):
            lams = xp.beta.sample((L,))
        else:
            lams = rng.beta(alpha, alpha, size=L)
        lams = xp.where(lams < 0.5, 1 - lams, lams)
        # Instead of direct advanced indexing assignment, use index_put_ for
        # torch:
        value = (
            lams[:, None] * arr_flat[donor1, batch_idx, :] +
            (1 - lams)[:, None] * arr_flat[donor2, batch_idx, :]
        )
        if is_torch(xp):
            arr_flat.masked_scatter_(mask[..., None], value)
        else:
            arr_flat[missing_rows, batch_idx, :] = value


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


def mean_diff(group1: Array, group2: Array,
              axis: int = -1, xp=None) -> np.ndarray[float] | float:
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
    >>> group1 = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]])
    >>> group2 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    >>> mean_diff(group1, group2, axis=1)
    array([ 0., 14.])
    >>> mean_diff(group1, group2, axis=0)
    array([ 0., 30.,  0.,  5.,  0.])
    >>> group3 = np.arange(100000, dtype=float).reshape(20000, 5)
    >>> mean_diff(group3, group1, axis=0)
    array([49997., 49968., 49999., 49995., 50001.])
    """

    if xp is None:
        xp = array_namespace(group1, group2)
    if is_numpy(xp):
        return _md(group1, group2, axes=[axis, axis])
    else:
        return group1.mean(axis=axis) - group2.mean(axis=axis)


if __name__ == "__main__":
    import numpy as np
    from timeit import timeit

    np.random.seed(0)
    n = 300
    group1 = np.random.rand(100, 100, 100)
    group2 = np.random.rand(500, 100, 100).astype('f2')
    group2[::2] = np.nan

    kwargs = dict(globals=globals(), number=n)
    time1 = timeit('mixup(group2.copy(), 0)', **kwargs)
    time2 = timeit('mixup3(group2.copy(), 0)', **kwargs)

    print(f"ttest: {time1 / n:.3g} per run")
    print(f"meandiff: {time2 / n:.3g} per run")
