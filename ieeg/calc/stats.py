import numpy as np
from joblib import Parallel, delayed
from mne.utils import logger
from skimage import measure

from ieeg import Doubles
from ieeg.calc.reshape import make_data_same
from scipy import stats as st
from numba import njit, guvectorize, float64


def weighted_avg_and_std(values, weights, axis=0):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, np.sqrt(variance) / np.sqrt(sum(weights) - 1))


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
    """ Calculate the average of data without trial outliers.

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
import mne    >>> import numpy as np
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
              axis: int | tuple[int] = None) -> np.ndarray | float:
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

    wh1 = ~np.isnan(group1)
    wh2 = ~np.isnan(group2)
    avg1 = np.mean(group1, axis=axis, where=wh1)
    avg2 = np.mean(group2, axis=axis, where=wh2)
    return avg1 - avg2


def window_averaged_shuffle(sig1: np.ndarray, sig2: np.ndarray,
                            n_perm: int = 1000, tails: int = 1,
                            obs_axis: int = 0, window_axis: int = -1,
                            stat_func: callable = mean_diff
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

    Returns
    -------
    shuffle_dist : np.ndarray
        The shuffle distribution.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> sig1 = np.array([[0,1,2,3,3,3,3,3,3,3,3,3,2,1,0]
    ... for _ in range(50)]) - rng.random((50, 15)) * 3.323
    >>> sig2 = np.array([[0] * 15 for _ in range(100)]) + rng.random((100, 15))
    >>> window_averaged_shuffle(sig1, sig2, n_perm=10000) #< 0.1
    True
    """

    # sig2 = make_data_same(sig2, sig1.shape, obs_axis, window_axis)

    # Concatenate the two signals for trial shuffling
    all_trial = np.concatenate((sig1, sig2), axis=obs_axis)
    labels = np.concatenate((np.full(sig1.shape[obs_axis], False, dtype=bool),
                             np.full(sig2.shape[obs_axis], True, dtype=bool)))

    # Calculate the observed difference
    obs_diff = stat_func(sig1, sig2, axis=(obs_axis, window_axis))
    if isinstance(obs_diff, tuple):
        logger.warn('Given stats function has more than one output. Accepting '
                    'only the first output')
        obs_diff = obs_diff[0]
        orig_func = stat_func

        def stat_func(s1, s2, axis):
            return orig_func(s1, s2, axis=axis)[0]

    # Calculate the difference between the two groups averaged across
    # trials and time
    diff = np.zeros((n_perm, *obs_diff.shape))
    for i in range(n_perm):
        perm_labels = np.random.permutation(labels)
        fake_sig1 = np.take(all_trial, np.where(np.invert(perm_labels))[0],
                            axis=obs_axis)
        fake_sig2 = np.take(all_trial, np.where(perm_labels)[0], axis=obs_axis)
        diff[i] = stat_func(fake_sig1, fake_sig2, axis=(obs_axis, window_axis))

    # Calculate the p-value
    p_act = np.mean(tail_compare(diff, obs_diff, tails), axis=0)

    return p_act


def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, p_thresh: float,
                      p_cluster: float = None, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0,
                      stat_func: callable = mean_diff,
                      ignore_adjacency: tuple[int] | int = None,
                      n_jobs: int = -1) -> np.ndarray[bool]:
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
        https://scipy.github.io/devdocs/reference/stats.html#independent-sample-tests
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 for all processors. Default
        is -1.

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
    >>> rng = np.random.default_rng(seed=42)
    >>> sig1 = np.array([[0,1,2,3,3,3,3,3,3,3,3,3,2,1,0]
    ... for _ in range(50)]) - rng.random((50, 15)) * 4
    >>> sig2 = np.array([[0] * 15 for _ in range(100)]) + rng.random((100, 15))
    >>> time_perm_cluster(sig1, sig2, 0.05, n_perm=10000)
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True, False, False, False])
    >>> time_perm_cluster(sig1, sig2, 0.01, n_perm=10000)
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True, False, False, False, False])
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
    p_act, diff = time_perm_shuffle(sig1, sig2, n_perm, tails, axis, True,
                                    stat_func)

    # Calculate the p value of the permutation distribution
    if tails == 1:
        p_perm = _perm_gt(diff)
    elif tails == 2:
        p_perm = _perm_gt(np.abs(diff))
    elif tails == -1:
        p_perm = _perm_lt(diff)
    else:
        raise ValueError('tails must be 1, 2, or -1')

    # Create binary clusters using the p value threshold
    b_act = tail_compare(1 - p_act, 1 - p_thresh, tails)
    b_perm = tail_compare(1 - p_perm, 1 - p_thresh, tails)

    # logger.info('Finding clusters')
    if ignore_adjacency is None:
        return time_cluster(b_act, b_perm, 1 - p_cluster, tails)

    # If there are axes to ignore, we need to loop over them
    clusters = np.zeros(b_act.shape, dtype=int)
    for i in np.ndindex(tuple(sig1.shape[i] for i in ignore_adjacency)):
        index = tuple(j for j in i) + (slice(None),)
        clusters[index] = time_cluster(
            b_act[index], b_perm[(slice(None),) + index], 1 - p_cluster, tails)

    return clusters


@guvectorize([(float64[:], float64[:])], '(n)->(n)', nopython=True)
def _perm_gt(diff, result):
    n = diff.shape[0]
    denom = n - 1
    for i in range(n):
        for j in range(n):
            if i != j and diff[i] > diff[j]:
                result[i] += 1
        result[i] /= denom


@guvectorize([(float64[:], float64[:])], '(n)->(n)', nopython=True)
def _perm_lt(diff, result):
    n = diff.shape[0]
    denom = n - 1
    for i in range(n):
        for j in range(n):
            if i != j and diff[i] < diff[j]:
                result[i] += 1
        result[i] /= denom


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
    act_clusters = measure.label(act, connectivity=1)
    perm_clusters = np.zeros(perm.shape, dtype=int)
    for i in range(perm.shape[0]):
        perm_clusters[i] = measure.label(perm[i], connectivity=1)

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
        larger = tail_compare(act_cluster_size, max_cluster_len, tails)
        cluster_p_values[act_cluster] = np.mean(larger, axis=0)

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
    tails.

    Parameters
    ----------
    diff : array, shape (..., time)
        The difference between the two groups.
    obs_diff : array, shape (..., time)
        The observed difference between the two groups.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.

    Returns
    -------
    larger : array, shape (..., time)
        The boolean array indicating whether the difference between the two
        groups is larger than the observed difference.
    """

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


def time_perm_shuffle(sig1: np.ndarray, sig2: np.ndarray, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0, return_perm: bool = False,
                      func: callable = mean_diff
                      ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Time permutation cluster test between two time series.

    The test is performed by shuffling the trials of the two time series and
    calculating the difference between the two groups at each time point. The
    p-value is the proportion of times the difference between the two groups
    is greater than the original observed difference. The number of trials in
    each group does not need to be the same.

    Parameters
    ----------
    sig1 : array, shape (trials, ..., time)
        Active signal. The first dimension is assumed to be the trials
    sig2 : array, shape (trials, ..., time)
        Passive signal. The first dimension is assumed to be the trials
    n_perm : int, optional
        The number of permutations to perform.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.
    axis : int, optional
        The axis to perform the permutation test across.
    return_perm : bool, optional
        If True, return the permutation distribution.
    func :
        The statistical function to use to compare populations. Requires an
        axis keyword input to denote observations (trials, for example).

    Returns
    -------
    p : np.ndarray, shape (..., time)
        The p-values for each time point.
        """
    # Concatenate the two signals for trial shuffling
    all_trial = np.concatenate((sig1, sig2), axis=axis)
    labels = np.concatenate((np.full(sig1.shape[axis], False, dtype=bool),
                             np.full(sig2.shape[axis], True, dtype=bool)))

    # Calculate the observed difference
    obs_diff = func(sig1, sig2, axis=axis)
    if isinstance(obs_diff, tuple):
        logger.warn('Given stats function has more than one output. Accepting '
                    'only the first output')
        obs_diff = obs_diff[0]
        orig_func = func

        def func(s1, s2, axis):
            return orig_func(s1, s2, axis=axis)[0]

    # Calculate the average difference between the two groups averaged across
    # trials at each time point
    diff = np.zeros((n_perm, *obs_diff.shape))
    for i in range(n_perm):
        perm_labels = np.random.permutation(labels)
        fake_sig1 = np.take(all_trial, np.where(np.invert(perm_labels))[0],
                            axis=axis)
        fake_sig2 = np.take(all_trial, np.where(perm_labels)[0], axis=axis)
        diff[i] = func(fake_sig1, fake_sig2, axis=axis)

    # Calculate the p-value
    p = np.mean(tail_compare(diff, obs_diff, tails), axis=0)

    if return_perm:
        return p, diff
    else:
        return p


@njit()
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


@njit()
def sine_f_test(window_fun: np.ndarray, x_p: np.ndarray
                ) -> (np.ndarray, np.ndarray):
    """computes the F-statistic for sine wave in locally-white noise.

    This function computes the F-statistic for a sine wave in locally-white
    noise. The sine wave is assumed to be of the form:
    .. math::
        x(t) = A \\sin(2 \\pi f t + \\phi)
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


if __name__ == '__main__':
    import numpy as np
    from timeit import timeit
    rng = np.random.default_rng(seed=42)
    sig1 = np.array([[0,1,2,3,3,3,3,3,3,3,3,3,2,1,0] for _ in range(50)]) - rng.random((50, 15)) * 4
    sig2 = np.array([[0] * 15 for _ in range(100)]) + rng.random((100, 15))
    p_act, diff = time_perm_shuffle(sig1, sig2, 10000, 1, 0, True)

    # Calculate the p value of the permutation distribution and compare
    # execution times

    p_perm1 = _perm_gt(diff)
    p_perm2 = np.sum(diff > diff[:, np.newaxis], axis=0) / (diff.shape[0] - 1)

    # Time the functions
    time2 = timeit('_perm_gt(diff)', globals=globals(), number=10)
    time1 = timeit('np.sum(diff > diff[:, np.newaxis], axis=0) / '
                   '(diff.shape[0] - 1)', globals=globals(), number=10)

    print(f'Time for calculate_p_perm: {time1:.6f} seconds')
    print(f'Time for _calculate_p_perm: {time2:.6f} seconds')
