import numpy as np
from joblib import Parallel, delayed
from mne.utils import logger
from scipy import stats as st
from scipy import ndimage
from scipy.ndimage import label

from ieeg import Doubles
from ieeg.calc.reshape import make_data_same
from ieeg.calc.cstats import mean_diff as _mean_diff, _perm_gt
from ieeg.process import get_mem
import psutil
from ieeg.calc.permgt import permgtnd


def dist(mat: np.ndarray, axis: int = 0, mode: str = 'sem',
         where: np.ndarray = None) -> Doubles:
    """ Calculate the mean and standard deviation of a matrix.

    This function calculates the mean and standard deviation of a matrix along
    a given axis. If a mask is provided, the mean and standard deviation are
    calculated only for the elements of the matrix that are not masked.

    Parameters
    ----------
    mat : np.ndarray
        Matrix to calculate mean and standard deviation of.
    axis : int
        Axis of matrix to calculate mean and standard deviation along.
    mode : str
        Mode of standard deviation to calculate. Can be 'sem' for standard
        error of the mean or 'std' for standard deviation.
    where : np.ndarray
        Mask of elements to include in mean and standard deviation calculation.

    Returns
    -------
    Doubles
        Tuple containing the mean and standard deviation of the matrix.

    Examples
    --------
    >>> import numpy as np
    >>> mat = np.arange(24).reshape(4,6)
    >>> dist(mat)[1] # doctest: +NORMALIZE_WHITESPACE
    array([3.87298335, 3.87298335, 3.87298335, 3.87298335, 3.87298335,
           3.87298335])
    >>> dist(mat, mode='std')[1]
    array([6.70820393, 6.70820393, 6.70820393, 6.70820393, 6.70820393,
           6.70820393])
    """

    assert mode in ('sem', 'std'), "mode must be 'sem' or 'std'"

    if where is None:
        where = np.ones(mat.shape, dtype=bool)

    where = np.logical_and(where, ~np.isnan(mat))

    mean = np.mean(mat, axis=axis, where=where)
    if mode == 'sem':
        std = st.sem(mat, axis=axis, nan_policy='omit')
    else:
        std = np.std(mat, axis=axis, where=where)

    return mean, std


def outlier_repeat(data: np.ndarray, sd: float, rounds: int = np.inf,
                   axis: int = 0) -> tuple[tuple[int, int]]:
    """ Remove outliers from data and repeat until no outliers are left.

    This function removes outliers from data and repeats until no outliers are
    left. Outliers are defined as any data point that is more than sd standard
    deviations from the mean. The function returns a tuple of tuples containing
    the index of the outlier and the round in which it was removed.

    Parameters
    ----------
    data : np.ndarray
        Data to remove outliers from.
    sd : float
        Number of standard deviations from the mean to consider an outlier.
    rounds : int
        Number of times to repeat outlier removal. If None, the function will
        repeat until no outliers are left.
    axis : int
        Axis of data to remove outliers from.

    Returns
    -------
    tuple[tuple[int, int]]
        Tuple of tuples containing the index of the outlier and the round in
        which it was removed.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]]).T
    >>> tuple(outlier_repeat(data, 1))
    ((1, 1), (3, 2))
    >>> tuple(outlier_repeat(data, 1, rounds=1))
    ((1, 1),)
    >>> tuple(outlier_repeat(data, 1, rounds=0))
    ()
    """
    inds = list(range(data.shape[axis]))

    # Square the data and set zeros to small positive number
    R2 = np.square(data)
    R2[np.where(R2 == 0)] = 1e-9

    # find all axes that are not channels (example: time, trials)
    axes = tuple(i for i in range(data.ndim) if not i == axis)

    # Initialize stats loop
    sig = np.std(R2, axes)  # take standard deviation of each channel
    cutoff = (sd * np.std(sig)) + np.mean(sig)  # outlier cutoff
    i = 1

    # remove bad channels and re-calculate variance until no outliers are left
    while np.any(np.where(sig > cutoff)) and i <= rounds:

        # Pop out names to bads output using comprehension list
        for j, out in enumerate(np.where(sig > cutoff)[0]):
            yield inds.pop(out - j), i

        # re-calculate per channel variance
        R2 = R2[..., np.where(sig < cutoff)[0], :]
        sig = np.std(R2, axes)
        cutoff = (sd * np.std(sig)) + np.mean(sig)
        i += 1


def find_outliers(data: np.ndarray, outliers: float) -> np.ndarray[bool]:
    """ Find outliers in data matrix.

    This function finds outliers in a data matrix. Outliers are defined as any
    trial with a maximum value greater than the mean plus outliers times the
    standard deviation. The function returns a boolean array with True for
    trials that are not outliers and False for trials that are outliers.

    Parameters
    ----------
    data : np.ndarray
        Data to find outliers in.
    outliers : float
        Number of standard deviations from the mean to consider an outlier.

    Returns
    -------
    np.ndarray[bool]
        Boolean array with True for trials that are not outliers and False
        for trials that are outliers.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]]).T
    >>> find_outliers(data, 1)
    array([ True, False,  True,  True,  True])
    >>> find_outliers(data, 3)
    array([ True,  True,  True,  True,  True])
    >>> find_outliers(data, 0.1)
    array([ True, False,  True, False,  True])
    """
    dat = np.abs(data)  # (trials X channels X (frequency) X time)
    max = np.max(dat, axis=-1)  # (trials X channels X (frequency))
    std = np.std(dat, axis=(-1, 0))  # (channels X (frequency))
    mean = np.mean(dat, axis=(-1, 0))  # (channels X (frequency))
    keep = max < ((outliers * std) + mean)  # (trials X channels X (frequency))
    return keep


def avg_no_outlier(data: np.ndarray, outliers: float = None,
                   keep: np.ndarray[bool] = None) -> np.ndarray:
    """Calculate the average of data without trial outliers.

    This function calculates the average of data without trial outliers.
    Outliers are defined as any trial with a maximum value greater than the
    mean plus outliers times the standard deviation.
    The function returns the average of data without outliers.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate average of.
    outliers : float
        Number of standard deviations from the mean to consider an outlier.
    keep : np.ndarray[bool]
        Boolean array with True for trials that are not outliers and False
        for trials that are outliers.

    Returns
    -------
    np.ndarray
        Average of data without outliers.

    Examples
    --------
    >>> import mne
    >>> mne.set_log_file(None)
    >>> data = np.array([[[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]]]).T
    >>> avg_no_outlier(data, 1)
    Removed Trial 0 in Channel 0
    Removed Trial 1 in Channel 0
    Removed Trial 1 in Channel 1
    Removed Trial 2 in Channel 0
    Removed Trial 3 in Channel 0
    Removed Trial 4 in Channel 0
    array([[nan],
           [2.5]])
    >>> avg_no_outlier(data, 3)
    Removed Trial 0 in Channel 0
    Removed Trial 1 in Channel 0
    Removed Trial 2 in Channel 0
    Removed Trial 3 in Channel 0
    Removed Trial 4 in Channel 0
    array([[nan],
           [14.]])
    """
    if data.ndim not in (3, 4):
        raise ValueError("Data must be 3D or 4D")

    if keep is None and outliers is not None:
        keep = find_outliers(data, outliers)
    elif keep is not None:
        if keep.ndim == 2 and data.ndim == 4:
            keep = keep[..., np.newaxis]
        elif keep.ndim == data.ndim:
            raise ValueError(f"Keep has too many dimensions ({keep.ndim})")
    else:
        raise ValueError("Either keep or outliers must be given")

    if np.squeeze(keep).ndim == 2:  # dat.ndim == 3
        disp = [f"Removed Trial {i} in Channel {j}" for i, j in np.ndindex(
            data.shape[0:2]) if not keep[i, j]]
    else:  # dat.ndim == 4:
        disp = [f"Removed Trial {i} in Channel {j} in Frequency {k}" for i, j,
                k in np.ndindex(data.shape[0:3]) if not keep[i, j, k]]
    for msg in disp:
        logger.info(msg)
    return np.mean(data, axis=0, where=keep[..., np.newaxis])


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

    return _mean_diff(in1, in2)


def window_averaged_shuffle(sig1: np.ndarray, sig2: np.ndarray,
                            n_perm: int = 1000, tails: int = 1,
                            obs_axis: int = 0, window_axis: int = -1,
                            stat_func: callable = mean_diff, seed: int = None,
                            ) -> np.ndarray[bool]:
    """Calculate the window averaged shuffle distribution.

    This function calculates the window averaged shuffle distribution for two
    groups of data. The shuffle distribution is calculated by randomly
    shuffling the data between the two groups and calculating the statistic
    function for each window, returning a distribution of the statistic
    corresponding to each window. The function returns the shuffle
    distribution.

    Parameters
    ----------
    sig1 : array, shape (trials, ..., time)
        The first group of observations.
    sig2 : array, shape (trials, ..., time)
        The second group of observations.
    n_perm : int, optional
        The number of permutations to perform. Default is 1000.
    tails : int, optional
        The number of tails to use for the p-value. Default is 1.
    obs_axis : int, optional
        The axis along which to calculate the statistic function. Default is 0.
    window_axis : int, optional
        The axis along which to calculate the window average. Default is -1.
    stat_func : callable, optional
        The statistic function to use. Default is mean_diff.
    seed : int, optional
        The random seed to use for the permutation test. Default is None.

    Returns
    -------
    shuffle_dist : np.ndarray
        The shuffle distribution.

    Examples
    --------
    >>> import numpy as np
    >>> seed = 43; rng = np.random.default_rng(seed)
    >>> sig1 = np.array([[0,1,1,2,2,2.5,3,3,3,2.5,2,2,1,1,0]
    ... for _ in range(50)])
    >>> sig2 = rng.random((100, 15)) * 3.2
    >>> window_averaged_shuffle(sig1, sig2, n_perm=10000, seed=seed)
    0.0002
    """

    # average the windows
    in1 = np.mean(sig1, axis=window_axis)
    in2 = np.mean(sig2, axis=window_axis)

    # calc obs axis
    obs_axis = obs_axis + sig1.ndim if obs_axis < 0 else obs_axis
    window_axis = window_axis + sig1.ndim if window_axis < 0 else window_axis
    obs_axis = obs_axis - 1 if window_axis < obs_axis else obs_axis

    # Calculate the observed difference
    obs_diff = stat_func(in1, in2, obs_axis)

    # Create shuffle distribution
    diff = shuffle_test(in1, in2, n_perm, obs_axis, stat_func, seed=seed)

    # Calculate the p-value
    p_act = np.mean(tail_compare(diff, obs_diff, tails), axis=0)

    return p_act


def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, p_thresh: float,
                      p_cluster: float = None, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0,
                      stat_func: callable = mean_diff,
                      ignore_adjacency: tuple[int] | int = None,
                      n_jobs: int = -1, seed: int = None) -> np.ndarray[bool]:
    """Calculate significant clusters using permutation testing and cluster
    correction.

    Takes two time series signals, finding clusters of activation defined as
    significant contiguous time points with a p value less than the p_thresh
    (greater if tails is -1, and both if tails is 2). The p value is the
    proportion of times that the difference in the statistic value for signal 1
    with respect to signal 2 is greater than the statistic for a random
    sampling of signal 1 and 2 with respect to a random sampling of signal 1
    and 2. The clusters are then corrected by only keeping clusters that are in
    the (1 - p_cluster)'th percentile of cluster lengths for signal 2.

    Parameters
    ----------
    sig1 : array, shape (trials, ..., time)
        Active signal. The first dimension is assumed to be the trials
    sig2 : array, shape (trials, ..., time)
        Passive signal. The first dimension is assumed to be the trials
    p_thresh : float
        The p-value threshold to use for determining significant time points.
    p_cluster : float, optional
        The p-value threshold to use for determining significant clusters.
    n_perm : int, optional
        The number of permutations to perform.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.
    axis : int, optional
        The axis to perform the permutation test across. Also known as the
        observations axis
    stat_func : callable, optional
        The statistical function to use to compare populations. Requires an
        axis keyword input to denote observations (trials, for example).
        Default function is `mean_diff`, but may be substituted with other test
        functions found here:
        https://scipy.github.io/devdocs/reference/stats.html#independent
        -sample-tests
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 for all processors. Default
        is -1.
    seed : int, optional
        The random seed to use for the permutation test. Default is None.

    Returns
    -------
    clusters : array, shape (..., time)
        The binary array of significant clusters.

    References
    ----------
    1. https://www.sciencedirect.com/science/article/pii/S0165027007001707

    Examples
    --------
    >>> import numpy as np
    >>> seed = 43; rng = np.random.default_rng(seed)
    >>> sig1 = np.array([[0,1,1,2,2,2.5,3,3,3,2.5,2,2,1,1,0]
    ... for _ in range(50)]) - rng.random((50, 15)) * 2.6
    >>> sig2 = np.array([[0] * 15 for _ in range(100)]) + rng.random(
    ... (100, 15))
    >>> time_perm_cluster(sig1, sig2, 0.05, n_perm=10000, seed=seed)
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True, False, False, False, False])
    >>> time_perm_cluster(sig1, sig2, 0.01, n_perm=10000, seed=seed)
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True, False, False, False, False, False])
    """
    # check inputs
    if p_cluster is None:
        p_cluster = p_thresh
    if tails not in [1, 2, -1]:
        raise ValueError('tails must be 1, 2, or -1')
    if p_cluster > 1 or p_cluster < 0 or p_thresh > 1 or p_thresh < 0:
        raise ValueError('p_thresh and p_cluster must be between 0 and 1')
    if isinstance(ignore_adjacency, int):
        ignore_adjacency = (ignore_adjacency,)
    if ignore_adjacency is not None:
        if axis in ignore_adjacency:
            raise ValueError('observations axis is eliminated before '
                             'clustering and so cannot be in ignore_adjacency')

        # axes where adjacency is ignored can be computed independently in
        # parallel
        out_shape = list(sig1.shape)
        out_shape.pop(axis)
        out = np.zeros(out_shape, dtype=int)
        ins = ((np.squeeze(sig1[:, i]), np.squeeze(sig2[:, i])) for i in
               np.ndindex(tuple(sig1.shape[j] for j in ignore_adjacency)))
        proc = Parallel(n_jobs, return_as='generator', verbose=40)(
            delayed(time_perm_cluster)(
                *i, p_thresh=p_thresh, p_cluster=p_cluster, n_perm=n_perm,
                tails=tails, axis=axis, stat_func=stat_func) for i in ins)
        for i, iout in enumerate(proc):
            out[i] = iout
        return out

    sig2 = make_data_same(sig2, sig1.shape, axis)

    # Calculate the p value of difference between the two groups
    # logger.info('Permuting events in shuffle test')
    # act = stat_func(sig1, sig2, axis=axis)
    # if isinstance(act, tuple):
    #     logger.warn('Given stats function has more than one output. Accepting '
    #                 'only the first output')
    #     orig_func = stat_func
    #
    #     def stat_func(s1, s2, axis):
    #         return orig_func(s1, s2, axis=axis)[0]
    # diff = shuffle_test(sig1, sig2, n_perm, axis, stat_func, seed=seed)
    #
    # # contatenate the actual group statistic and concatenate with the null
    # # distribution along the observations axis
    #
    # act = np.expand_dims(act, axis=axis)
    # p_act = np.mean(tail_compare(diff, act, tails), axis=0)
    if tails == 1:
        alt = 'greater'
    elif tails == -1:
        alt = 'less'
    else:
        alt = 'two-sided'
    out_mem = (sig1.size + sig2.size) * 8
    batch_size = get_mem() // out_mem
    res = st.permutation_test([sig1, sig2], stat_func,
                              n_resamples=n_perm,
                              alternative=alt,
                              batch=batch_size)
    p_act = res.pvalue
    diff = res.null_distribution

    # all_diff = np.concatenate((act, diff), axis=axis)

    # Calculate the p value of the permutation distribution
    # p_act = proportion(act, diff, tails, axis=0)
    p_perm = proportion(diff, tail=tails, axis=0)

    # Create binary clusters using the p value threshold
    b_act = tail_compare(1. - p_act, 1. - p_thresh, tails)
    b_perm = tail_compare(1. - p_perm, 1. - p_thresh, tails)

    # logger.info('Finding clusters')
    if ignore_adjacency is None:
        return time_cluster(b_act, b_perm, 1 - p_cluster, tails)

    # If there are axes to ignore, we need to loop over them
    clusters = np.zeros(b_act.shape, dtype=int)
    for i in np.ndindex(tuple(sig1.shape[i] for i in ignore_adjacency)):
        index = tuple(j for j in i) + (slice(None),)
        clusters[index] = time_cluster(
            b_act[index], b_perm[(slice(None),) + index], 1 - p_cluster, tails)

    return clusters, 1. - p_act


def proportion(val: np.ndarray[float, ...] | float,
               comp: np.ndarray[float, ...] = None, tail: int = 1,
               axis: int = None) -> np.ndarray[float, ...] | float:
    """takes a value and a comparison and returns the proportion of the
    comparison that is greater than the value

    Parameters
    ----------
    val : array, shape (x, ...) or float,
        The difference between two groups.
    comp : array, shape (y, ...) optional
        The difference between two groups.
    tail : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.
    axis : int, optional
        The axis to perform the permutation test across. Also known as the
        observations axis

    Returns
    -------
    proportion : array, shape (..., time)
        The proportion of the comparison that is greater than the value.

    Examples
    ________
    >>> import numpy as np
    >>> rand = np.random.default_rng(seed=42)
    >>> diff1 = rand.random(5)
    >>> diff1
    array([0.77395605, 0.43887844, 0.85859792, 0.69736803, 0.09417735])
    >>> proportion(diff1)
    array([0.75, 0.25, 1.  , 0.5 , 0.  ])
    >>> np.sum(diff1 > diff1[:, None], axis=0) / (diff1.shape[0] - 1)
    array([0.75, 0.25, 1.  , 0.5 , 0.  ])
    >>> diff2 = rand.random((2, 4))
    >>> diff2
    array([[0.97562235, 0.7611397 , 0.78606431, 0.12811363],
           [0.45038594, 0.37079802, 0.92676499, 0.64386512]])
    >>> proportion(diff2, axis=1)
    array([[1.        , 0.33333333, 0.66666667, 0.        ],
           [0.33333333, 0.        , 1.        , 0.66666667]])
    >>> proportion(diff2, axis=0)
    array([[1., 1., 0., 0.],
           [0., 0., 1., 1.]])
    >>> val = 0.5
    >>> compare = np.array([0.2, 0.4, 0.5, 0.7, 0.9])
    >>> proportion(val, compare)
    0.4
    >>> proportion(compare, compare) * compare.shape[0] / (
    ... compare.shape[0] - 1)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> proportion(compare)
    >>> val = np.full(5, 0.5)
    >>> compare = np.array([[0.2, 0.4, 0.4, 0.7, 0.9],
    ...                     [0.1, 0.3, 0.6, 0.5, 0.9]])
    >>> proportion(val, compare, axis=0)
    array([1. , 1. , 0.5, 0. , 0. ])
    >>> proportion(compare, compare[:, None], axis=0
    ... ) * compare.shape[0] / (compare.shape[0] - 1)
    array([[1., 1., 0., 1., 0.],
           [0., 0., 1., 0., 0.]])
    """

    match tail:
        case 1:
            pass
        case 2:
            val = np.abs(val)
            if comp is not None:
                comp = np.abs(comp)
        case -1:
            val *= -1
            if comp is not None:
                comp *= -1
        case _:
            raise ValueError('tail must be 1, 2, or -1')

    if axis is None and comp is None:
        return _perm_gt_1d(val)
    elif comp is None:
        return _perm_gt_1d(val, axis=axis)
    else:
        raise NotImplementedError()


# @guvectorize(['void(f8[::1], f8[::1])'], '(n)->(n)', nopython=True)
def _perm_gt_1d(diff, axis=0):
    m = diff.shape[axis] - 1
    sorted_indices = diff.argsort(axis=axis)  # Get sorted indices
    proportions = np.arange(diff.shape[axis]) / m  # Create proportions array
    # Rearrange to match original order
    return proportions[sorted_indices.argsort(axis=axis)]


def time_cluster(act: np.ndarray, perm: np.ndarray, p_val: float = None,
                 tails: int = 1) -> np.ndarray:
    """Cluster correction for time series data.

    1. Creates an index of all the binary clusters in the active and permuted
        passive data.
    2. For each permutation in the passive data, determine the maximum cluster
        size.
    3. For each cluster in the active data, determine the proportion of
        permutations that have a cluster of the same size or larger.

    Parameters
    ----------

    act : array, shape (time)
        The active data.
    perm : array, shape (n_perm, time)
        The permutation passive data.
    p_val : float, optional
        The p-value threshold to use for determining significant clusters.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.

    Returns
    -------
    clusters : array, shape (..., time)
        The clusters.
    """

    # Create an index of all the binary clusters in the active and permuted
    # passive data
    act_clusters = label(act)
    perm_clusters = np.zeros(perm.shape, dtype=int)
    for i in range(perm.shape[0]):
        perm_clusters[i] = label(perm[i])

    # For each permutation in the passive data, determine the maximum cluster
    # size
    max_cluster_len = np.zeros(perm_clusters.shape[0])
    for i in range(perm_clusters.shape[0]):
        for j in range(1, perm_clusters.max() + 1):
            max_cluster_len[i] = np.maximum(max_cluster_len[i], np.sum(
                perm_clusters[i] == j))

    # For each cluster in the active data, determine the proportion of
    # permutations that have a cluster of the same size or larger
    cluster_p_values = np.zeros(act_clusters.shape)
    for i in range(1, act_clusters.max() + 1):
        # Get the cluster
        act_cluster = act_clusters == i
        # Get the cluster size
        act_cluster_size = np.sum(act_cluster)
        # Determine the proportion of permutations that have a cluster of the
        # same size or larger
        cluster_p_values[act_cluster] = np.mean(act_cluster_size >
                                                max_cluster_len, axis=0)

    # If p_val is not None, return the boolean array indicating whether the
    # cluster is significant
    if p_val is not None:
        return tail_compare(cluster_p_values, p_val, tails)
    else:
        return cluster_p_values


def tail_compare(diff: np.ndarray | float | int,
                 obs_diff: np.ndarray | float | int, tails: int = 1
                 ) -> np.ndarray | bool:
    """Compare the difference between two groups to the observed difference.

    This function applies the appropriate comparison based on the number of
    tails. The shapes of the two arrays must be broadcastable.

    Parameters
    ----------
    diff : array
        The difference between the two groups.
    obs_diff : array
        The observed difference between the two groups.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.

    Returns
    -------
    larger : array, shape (..., time)
        The boolean array indicating whether the difference between the two
        groups is larger than the observed difference.

    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.default_rng(seed=42)
    >>> diff1 = rand.random(5) - 0.5
    >>> diff1
    array([ 0.27395605, -0.06112156,  0.35859792,  0.19736803, -0.40582265])
    >>> obs_diff1 = 0.25
    >>> tail_compare(diff1, obs_diff1)
    array([ True, False,  True, False, False])
    >>> tail_compare(diff1, obs_diff1, tails=2)
    array([ True, False,  True, False,  True])
    >>> tail_compare(diff1, obs_diff1, tails=-1)
    array([False,  True, False,  True,  True])
    >>> tail_compare(diff1, np.array([1, 2])
    ... ) # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: shape mismatch: objects cannot be broadcast to a single ...
    """

    # check if arrays are broadcastable
    try:
        np.broadcast(diff, obs_diff)
    except ValueError as e:
        raise e

    # Account for one or two tailed test
    match tails:
        case 1:
            temp = np.greater(diff, obs_diff)
        case 2:
            temp = np.greater(np.abs(diff), np.abs(obs_diff))
        case -1:
            temp = np.less(diff, obs_diff)
        case _:
            raise ValueError('tails must be 1, 2, or -1')

    return temp


def shuffle_test(sig1: np.ndarray, sig2: np.ndarray, n_perm: int = 1000,
                 axis: int = 0, func: callable = mean_diff, seed: int = None,
                 n_jobs: int = -3) -> np.ndarray:
    """Time permutation shuffle test between two set of observations.

    The test is performed by shuffling the trials and calculating the test
    statistic (the group difference by default). The shuffling together of
    groups should create a null distribution of values to compare against for
    statistical significance. The number of trials in each group does not need
    to be the same, but all other dimensions must be.

    Parameters
    ----------
    sig1 : array, shape (trials1, ...)
        Active signal. The first dimension is assumed to be the trials
    sig2 : array, shape (trials2, ...)
        Passive signal. The first dimension is assumed to be the trials
    n_perm : int, optional
        The number of permutations to perform.
    axis : int, optional
        The axis to perform the permutation test across (trials).
    func :
        The statistical function to use to compare populations. Requires an
        axis keyword input to denote observations (trials, for example).
    seed : int, optional
        The seed for the random number generator.
    n_jobs : int, optional
        The number of jobs to run in parallel. Only used if the permutation
        test will exceed memory. Default is -3.

    Returns
    -------
    p : np.ndarray, shape (...)
        The p-values

    Examples
    --------
    >>> import numpy as np
    >>> seed = 42; rng = np.random.default_rng(seed)
    >>> sig1 = np.mean(np.array([[0,1,1,2,2,2.5,3,3,3,2.5,2,2,1,1,0]]), axis=1)
    >>> sig2 = np.mean(rng.random((100, 15)) * 2.4, axis=1)
    >>> round(np.mean(sig1 - sig2), 3)
    0.534
    >>> np.mean(np.mean(sig1 - sig2) > shuffle_test(sig1, sig2, n_perm=1000000,
    ... seed=seed))
    0.993
    """

    rng = np.random.default_rng(seed)
    axis = axis + sig1.ndim if axis < 0 else axis

    # Concatenate the two signals for trial shuffling
    all_trial = np.concatenate((sig1, sig2), axis=axis)
    shape = sig1.shape

    # Generate all permutations at once
    idx = np.tile(np.arange(all_trial.shape[axis]), (n_perm, 1))
    rng.permuted(idx, axis=1, out=idx)
    idx1 = idx[:, :shape[axis]]
    idx2 = idx[:, shape[axis]:]

    # Check if the permutation will exceed memory
    out_mem = all_trial.size * n_perm * 8
    if out_mem > get_mem() * psutil.cpu_count():
        logger.warning(f"Permutation test will exceed memory ({out_mem} bytes)"
                       f", using a generator instead. This may take longer.")

        def _shuffle_test(idx_1, idx_2):
            fake_sig1 = np.take(all_trial, idx_1, axis=axis)
            fake_sig2 = np.take(all_trial, idx_2, axis=axis)
            return func(fake_sig1, fake_sig2, axis=axis)

        diff = np.zeros((n_perm, *shape[:axis], *shape[axis + 1:]))
        proc = Parallel(n_jobs=n_jobs, verbose=40)(delayed(_shuffle_test)(
            idx1[i], idx2[i]) for i in range(n_perm))
        for i, out in enumerate(proc):
            diff[i] = out
    else:

        # Split the permuted trials into fake_sig1 and fake_sig2
        fake_sig1 = np.take(all_trial, idx1, axis=axis)
        fake_sig2 = np.take(all_trial, idx2, axis=axis)

        # Calculate the average difference between the two groups averaged
        # across trials
        diff = func(fake_sig1, fake_sig2, axis=axis + 1)

    return diff


def sum_squared(x: np.ndarray) -> np.ndarray | float:
    """Compute norm of an array.

    Parameters
    ----------
    x : array
        Data whose norm must be found.
    Returns
    -------
    value : float
        Sum of squares of the input array x.
    """
    x_flat = x.ravel()
    return np.dot(x_flat, x_flat)


def sine_f_test(window_fun: np.ndarray, x_p: np.ndarray
                ) -> (np.ndarray, np.ndarray):
    """Computes the F-statistic for sine wave in locally-white noise.

    This function computes the F-statistic for a sine wave in locally-white
    noise. The sine wave is assumed to be of the form:

    :math:`x(t) = A \\sin(2 \\pi f t + \\phi)`

    where :math:`A` is the amplitude of the sine wave, :math:`f` is the
    frequency of the sine wave, and :math:`\\phi` is the phase of the sine
    wave. The F-statistic is computed by taking the ratio of the variance of
    the sine wave to the variance of the noise. The variance of the sine wave
    is computed by taking the sum of the squares of the sine wave across tapers
    and then dividing by the number of tapers. The variance of the noise is
    computed by taking the sum of the squares of the residuals across tapers
    and then dividing by the number of tapers minus one. The F-statistic is
    then computed by taking the ratio of the variance of the sine wave to the
    variance of the noise.

    Parameters
    ----------
    window_fun : array
        The tapers used to calculate the multitaper spectrum.
    x_p : array
        The tapered time series.

    Returns
    -------
    f_stat : array
        The F-statistic for each frequency.
    A : array
        The amplitude of the sine wave at each frequency.

    Examples:
    ---------
    >>> import numpy as np
    >>> window_fun = np.array([[0.5, 0.5], [0.5, -0.5]])
    >>> x_p = np.array([[[1, 1, 1, 1, 1], [0, 60, 0, 10, 0]],
    ...                 [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]]).T
    >>> sine_f_test(window_fun, x_p)
    (array([[0.        , 0.        ],
           [0.00027778, 0.        ],
           [0.        , 0.        ],
           [0.01      , 0.        ],
           [0.        , 0.        ]]), array([[1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.]]))

    """
    # drop the even tapers
    n_tapers = len(window_fun)
    tapers_odd = np.arange(0, n_tapers, 2)
    tapers_even = np.arange(1, n_tapers, 2)
    tapers_use = window_fun[tapers_odd]

    # sum tapers for (used) odd prolates across time (n_tapers, 1)
    H0 = np.sum(tapers_use, axis=1)

    # sum of squares across tapers (1, )
    H0_sq = sum_squared(H0)

    # sum of the product of x_p and H0 across tapers (1, n_freqs)
    exp_H0 = np.reshape(H0, (1, -1, 1))
    x_p_H0 = np.sum(x_p[:, tapers_odd, :] * exp_H0, axis=1)

    # resulting calculated amplitudes for all freqs
    A = x_p_H0 / H0_sq

    # figure out which freqs to remove using F stat

    # estimated coefficient
    x_hat = A * np.reshape(H0, (-1, 1))

    # numerator for F-statistic
    num = (n_tapers - 1) * (A * A.conj()).real * H0_sq
    # denominator for F-statistic
    den = (np.sum(np.abs(x_p[:, tapers_odd, :] - x_hat) ** 2, 1) +
           np.sum(np.abs(x_p[:, tapers_even, :]) ** 2, 1))
    den = np.where(den == 0, np.inf, den)
    f_stat = num / den

    return f_stat, A


def label(label_image, background=None, return_num=False, connectivity=1):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    label_image : ndarray of dtype int
        Image to label.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.
        Default is 1 for ieeg usage.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    regionprops
    regionprops_table

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3).astype(bool)
    >>> print(x)
    [[ True False False]
     [False  True False]
     [False False  True]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=False))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, background=True))
    [[0 1 1]
     [2 0 1]
     [2 2 0]]
    """
    if background == 1:
        label_image = ~label_image

    if connectivity is None:
        connectivity = label_image.ndim

    if not 1 <= connectivity <= label_image.ndim:
        raise ValueError(
            f'Connectivity for {label_image.ndim}D label_image should '
            f'be in [1, ..., {label_image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood(None, connectivity, label_image.ndim)
    result = ndimage.label(label_image, structure=footprint)

    if return_num:
        return result
    else:
        return result[0]


def _resolve_neighborhood(footprint, connectivity, ndim):
    """Validate or create a footprint (structuring element).

    Depending on the values of `connectivity` and `footprint` this function
    either creates a new footprint (`footprint` is None) using `connectivity`
    or validates the given footprint (`footprint` is not None).

    Parameters
    ----------
    footprint : ndarray
        The footprint (structuring) element used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    ndim : int
        Number of dimensions `footprint` ought to have.
    enforce_adjacency : bool
        A boolean that determines whether footprint must only specify direct
        neighbors.

    Returns
    -------
    footprint : ndarray
        Validated or new footprint specifying the neighborhood.

    Examples
    --------
    >>> _resolve_neighborhood(None, 1, 2)
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]])
    >>> _resolve_neighborhood(None, None, 3).shape
    (3, 3, 3)
    """
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        footprint = ndimage.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        footprint = np.asarray(footprint, dtype=bool)
        # Must specify neighbors for all dimensions
        if footprint.ndim != ndim:
            raise ValueError(
                "number of dimensions in image and footprint do not" "match"
            )
        # Must only specify direct neighbors
        if any(s != 3 for s in footprint.shape):
            raise ValueError("dimension size in footprint is not 3")
        elif any((s % 2 != 1) for s in footprint.shape):
            raise ValueError("footprint size must be odd along all dimensions")

    return footprint


if __name__ == '__main__':
    import numpy as np
    from timeit import timeit

    rng = np.random.default_rng(seed=42)
    sig1 = np.array([[0, 1, 2, 3, 3] for _ in range(50)]) - rng.random(
        (50, 5)) * 5
    sig2 = np.array([[0] * 5 for _ in range(100)]) + rng.random((100, 5))
    diff = shuffle_test(sig1, sig2, 3000, 0)
    act = mean_diff(sig1, sig2, axis=0)

    # Calculate the p value of the permutation distribution and compare
    # execution times

    # p_perm1 = _perm_gt(diff, diff)
    p_perm2 = np.sum(diff[None] > diff[:, None], axis=0) / (diff.shape[0] - 1)
    p_perm3 = (_perm_gt(diff, diff[:, None], axis=0) * diff.shape[0] /
               (diff.shape[0] - 1))
    p_perm4 = proportion(diff, axis=0)

    # Time the functions
    runs = 20
    # time1 = timeit('_perm_gt(diff, diff)', globals=globals(), number=runs)
    time2 = timeit('np.sum(diff > diff[:, np.newaxis], axis=0) / '
                   '(diff.shape[0] - 1)', globals=globals(), number=runs)
    time3 = timeit('_perm_gt(diff[:, None], diff, axis=0) * diff.shape[0]'
                   '/ (diff.shape[0] - 1)', globals=globals(), number=runs)
    time4 = timeit('proportion(diff, axis=0)', globals=globals(), number=runs)

    # print(f'Time for _perm_gt_2: {time1 / runs:.6f} seconds per run')
    print(f'Time for sum method: {time2 / runs:.6f} seconds per run')
    print(f'Time for _perm_gt: {time3 / runs:.6f} seconds per run')
    print(f'Time for perm_gt: {time4 / runs:.6f} seconds per run')
