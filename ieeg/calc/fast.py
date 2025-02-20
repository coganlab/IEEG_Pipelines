import numpy as np
from ieeg.calc._fast.ufuncs import mean_diff as _md, t_test as _ttest
from ieeg.calc._fast.mixup import mixupnd as cmixup, normnd as cnorm
from ieeg.calc._fast.permgt import permgtnd as permgt
from ieeg.arrays.api import array_namespace, is_numpy, is_torch, Array
from scipy.stats import rankdata
from functools import partial

__all__ = ["mean_diff", "mixup", "permgt", "norm", "concatenate_arrays",
           "ttest", "brunnermunzel"]

def brunnermunzel(x: np.ndarray, y: np.ndarray, axis=None, nan_policy='omit'):
    """Compute the Brunner-Munzel test on samples x and y.

    The Brunner-Munzel test is a nonparametric test of the null hypothesis that
    when values are taken one by one from each group, the probabilities of
    getting large values in both groups are equal.
    Unlike the Wilcoxon-Mann-Whitney's U test, this does not require the
    assumption of equivariance of two groups. Note that this does not assume
    the distributions are same. This test works on two independent samples,
    which may have different sizes.

    Parameters
    ----------
    x, y : array_like
        Array of samples, should be one-dimensional.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    distribution : {'t', 'normal'}, optional
        Defines how to get the p-value.
        The following options are available (default is 't'):

          * 't': get the p-value by t-distribution
          * 'normal': get the p-value by standard normal distribution.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.

    See Also
    --------
    mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_).

    References
    ----------
    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal. Vol. 42(2000): 17-25.
    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the
           non-parametric Behrens-Fisher problem". Computational Statistics and
           Data Analysis. Vol. 51(2007): 5192-5204.

    Examples
    --------
    >>> from scipy.stats import brunnermunzel as bz
    >>> x1 = np.array([1,2,1,1,1,1,1,1,1,1,2,4,1,1])
    >>> x2 = np.array([3,3,4,3,1,2,3,1,1,5,4])
    >>> brunnermunzel(x1, x2), bz(x1, x2, alternative='greater').statistic
    3.1374674823029505
    >>> x3 = np.array([[1,2,1,1],[1,1,1,1],[1,1,2,4]])
    >>> x4 = np.array([[3,3,4,3],[1,2,3,1], [1,5,4,4]])
    >>> brunnermunzel(x3, x4, axis=0), bz(x3, x4, axis=0, alternative='greater').statistic
    3.1374674823029505
    >>> brunnermunzel(x3, x4, axis=1), bz(x3, x4, axis=1, alternative='greater').statistic
    >>> brunnermunzel(x3, x4, axis=None), bz(x3, x4, axis=None, alternative='greater').statistic
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
          axis: int) -> np.ndarray:
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
    >>> group3 = np.arange(100000000, dtype=float).reshape(200000, 500)
    >>> ttest(group3, group1.repeat(100, 1), 0)
    array([244.92741947, 242.26926888, 244.93721715, 244.858866  ,
           244.94701484])
    """
    return _ttest(group1, group2, axes=[axis, axis])

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
    >>> concatenate_arrays((arr1, arr2), axis=0)
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
    >>> concatenate_arrays((arr2[0].astype('f2'), arr1[0].astype('f2')), axis=1)
    array([[ 0.,  1.,  2.,  3.,  0.,  1.,  2.],
           [ 4.,  5.,  6.,  7.,  3.,  4.,  5.],
           [ 8.,  9., 10., 11., nan, nan, nan]], dtype=float16)
    """

    if axis is None:
        axis = 0
        arrays = [np.expand_dims(ar, axis) for ar in arrays]

    arrays = [ar.astype(float) if ar.dtype.kind in 'iu' else ar for ar in arrays ]

    while axis < 0:
        axis += max(a.ndim for a in arrays)

    max_shape = [max(a.shape[ax] for a in arrays) if ax!=axis else
                 sum(a.shape[ax] for a in arrays) for ax in range(arrays[0].ndim)]
    out = np.full(max_shape, np.nan, dtype=arrays[0].dtype)
    start = 0
    for i, ar in enumerate(arrays):
        slices = tuple(slice(start, start + ar.shape[ax]) if ax == axis else slice(ar.shape[ax])
                       for ax in range(ar.ndim))
        out[*slices] = ar
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

    if arr.dtype != np.float64:
        temp = arr.astype(float, copy=True)
        _mixup_np(temp, obs_axis, alpha, rng)
        arr[...] = temp
        return

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

        # temp = arr.astype(float, copy=True)
        cmixup(arr, 1, alpha, rng)
        # arr[...] = temp

def mixup(arr: Array, obs_axis: int, alpha: float = 1.,
                     rng=None) -> None:
    """
    Replace rows along the observation axis that are “missing” (i.e. contain any NaNs)
    with a random convex combination of two non‐missing rows (the “mixup”).

    This function works for arrays of arbitrary dimension so long as the
    observation axis (obs_axis) contains the “rows” to mix up and the last axis
    holds features. In higher dimensions the axes other than obs_axis and the
    last axis are treated as independent batch indices. (Every such batch is assumed
    to have at least one non-NaN row.)

    The mixup coefficient for each missing row is drawn from a beta distribution
    with parameters (alpha, alpha) and then “flipped” if it is less than 0.5 (so that
    the coefficient is always >=0.5).

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
        A random number generator (if None, one is created using np.random.RandomState()).

    Returns
    -------
    None; arr is modified in-place.

    Examples
    --------
    >>> np.random.seed(0)
    >>> arr = np.array([[1, 2],
    ...                 [4, 5],
    ...                 [7, 8],
    ...                 [float("nan"), float("nan")]])
    >>> mixup(arr, 0)
    >>> arr
    array([[1.        , 2.        ],
           [4.        , 5.        ],
           [7.        , 8.        ],
           [6.03943491, 7.03943491]])

    For a 3D example (here we mix along axis 1):
    >>> arr3 = np.arange(24, dtype=float).reshape(2, 3, 4)
    >>> arr3[0, 2, :] = [float("nan")] * 4
    >>> mixup(arr3, 1)
    >>> arr3
    ... # arr3[0,2,:] has been replaced with a mixup of two non-NaN rows from arr3[0,:,:]
    >>> group2 = np.random.rand(500, 10, 10, 100).astype("float16")
    >>> group2[::2, 0, 0, :] = np.nan
    >>> mixup(group2, 0)
    >>> group2[:10, 0, 0, :5]
    >>> import cupy as cp
    >>> group3 = cp.randn(100, 10, 10, 100)
    >>> group3[0::2, 0, 0, :] = float("nan")
    >>> mixup(group3, 0)
    >>> group3[0, 0, :, :5]
    """
    xp = array_namespace(arr)
    if is_numpy(xp):
        _mixup_np(arr, obs_axis, alpha, rng)
        return
    elif is_torch(xp): # TODO: remove this crutch to keep data on the GPU
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
    ndim = arr_view.ndim

    if ndim == 2:
        # 2D case: shape is (n_obs, n_features)
        mask = xp.isnan(arr_view).any(axis=-1)  # True for rows with any NaN
        missing_idx = xp.nonzero(mask)[0]
        if missing_idx.size:
            non_missing_idx = xp.nonzero(~mask)[0]
            # For every missing row, randomly choose two non-missing donor rows.
            donors = rng.choice(non_missing_idx, size=(missing_idx.size, 2))
            # Draw mixing coefficients from Beta(alpha, alpha) and flip if < 0.5.
            lams = rng.beta(alpha, alpha, size=missing_idx.size)
            lams = xp.where(lams < 0.5, 1 - lams, lams)
            # Replace the missing rows with a convex combination.
            arr_view[missing_idx] = (
                lams[:, None] * arr_view[donors[:, 0]] +
                (1 - lams)[:, None] * arr_view[donors[:, 1]]
            )
    else:
        # For ndim >= 3, assume that the last axis holds features.
        # Flatten all intermediate (batch) dimensions into one.
        n_obs = arr_view.shape[0]
        n_features = arr_view.shape[-1]
        if is_torch(xp) and not arr_view.is_contiguous():
            arr_view = arr_view.contiguous()

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
            # For each missing observation, generate a random index into the available
            # (non-missing) rows in its batch.
            idx1 = xp.astype(rng.rand(L) * counts[batch_idx], int)
            idx2 = xp.astype(rng.rand(L) * counts[batch_idx], int)
            donor1 = order[idx1, batch_idx]
            donor2 = order[idx2, batch_idx]
            if is_torch(xp):
                lams = xp.beta.sample((L,))
            else:
                lams = rng.beta(alpha, alpha, size=L)
            lams = xp.where(lams < 0.5, 1 - lams, lams)
            # Instead of direct advanced indexing assignment, use index_put_ for torch:
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
    return _md(group1, group2, axes=[axis, axis])


if __name__ == "__main__":
    import numpy as np
    from timeit import timeit

    np.random.seed(0)
    n = 300
    group1 = np.random.rand(100, 100, 100)
    group2 = np.random.rand(500, 100, 100)
    group2[::2] = np.nan

    kwargs = dict(globals=globals(), number=n)
    time1 = timeit('mixup(group2.copy(), 0)', **kwargs)
    time2 = timeit('mixup3(group2.copy(), 0)', **kwargs)

    print(f"ttest: {time1 / n:.3g} per run")
    print(f"meandiff: {time2 / n:.3g} per run")