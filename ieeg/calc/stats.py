import numpy as np
from functools import partial

from skimage import measure
from mne.utils import logger

from ieeg import Doubles
from ieeg.process import proc_array


def dist(mat: np.ndarray, mask: np.ndarray = None, axis: int = 0) -> Doubles:
    """ Calculate the mean and standard deviation of a matrix.

    This function calculates the mean and standard deviation of a matrix along
    a given axis. If a mask is provided, the mean and standard deviation are
    calculated only for the elements of the matrix that are not masked.

    Parameters
    ----------
    mat : np.ndarray
        Matrix to calculate mean and standard deviation of.
    mask : np.ndarray
        Mask to apply to matrix before calculating mean and standard deviation.
    axis : int
        Axis of matrix to calculate mean and standard deviation along.

    Returns
    -------
    Doubles
        Tuple containing the mean and standard deviation of the matrix.
    """

    if mask is None:
        mask = np.ones(mat.shape)
        mask[np.isnan(mask)] = 0
    elif mat.shape != mask.shape:
        raise ValueError(f"matrix shape {mat.shape} not same as mask shape "
                         f"{mask.shape}")
    mask[np.isnan(mask)] = 0

    avg = np.divide(np.sum(np.multiply(mat, mask), axis), np.sum(mask, axis))
    avg = np.reshape(avg, [np.shape(avg)[axis]])
    stdev = np.std(mat, axis) / np.sqrt(np.shape(mat)[axis + 1])
    stdev = np.reshape(stdev, [np.shape(stdev)[axis]])
    return avg, stdev


def outlier_repeat(data: np.ndarray, sd: float, rounds: int = None,
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
        for trials that are outliers."""
    dat = np.abs(data)  # (trials X channels X (frequency) X time)
    max = np.max(dat, axis=-1)  # (trials X channels X (frequency))
    std = np.std(dat, axis=(-1, 0))  # (channels X (frequency))
    mean = np.mean(dat, axis=(-1, 0))  # (channels X (frequency))
    keep = max < ((outliers * std) + mean)  # (trials X channels X (frequency))
    return keep


def avg_no_outlier(data: np.ndarray, outliers: float = None,
                   keep: np.ndarray[bool] = None) -> np.ndarray:
    """ Calculate the average of data without trial outliers.

    This function calculates the average of data without trial outliers. Outliers
    are defined as any trial with a maximum value greater than the mean plus
    outliers times the standard deviation. The function returns the average of
    data without outliers.
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
    """ Calculate the mean difference between two groups.

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
    """

    avg1 = np.nanmean(group1, axis=axis)
    avg2 = np.nanmean(group2, axis=axis)

    return avg1 - avg2


def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, p_thresh: float,
                      p_cluster: float = None, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0,
                      stat_func: callable = mean_diff,
                      ignore_adjacency: tuple[int] | int = None) -> np.ndarray:
    """ Calculate significant clusters using permutation testing and cluster
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

    Returns
    -------
    clusters : array, shape (..., time)
        The binary array of significant clusters.

    References
    ----------
    1. https://www.sciencedirect.com/science/article/pii/S0165027007001707
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
    elif ignore_adjacency is not None:
        if axis in ignore_adjacency:
            raise ValueError('observations axis is eliminated before '
                             'clustering and so cannot be in ignore_adjacency')

    # Make sure the data is the same shape
    eq = list(np.equal(sig1.shape, sig2.shape)[np.arange(sig1.ndim) != axis])
    if not all(eq):
        eq.insert(axis, True)
        pad_shape = [(0, 0) if eq[i] else
                     (0, sig1.shape[i] - sig2.shape[i])
                     for i in range(sig1.ndim)]
        sig2 = np.pad(sig2, pad_shape, mode='reflect')

    # Calculate the p value of difference between the two groups
    # logger.info('Permuting events in shuffle test')
    p_act, diff = time_perm_shuffle(sig1, sig2, n_perm, tails, axis, True,
                                    stat_func)

    # Calculate the p value of the permutation distribution
    # logger.info('Calculating permutation distribution')
    # p_perm = np.zeros(diff.shape, dtype=np.float16)
    # for i in range(diff.shape[0]):
    #     # p_perm is the probability of observing a difference as large as the
    #     # other permutations, or larger, by chance
    #
    #     larger = tail_compare(diff[i], diff[np.arange(len(diff)) != i], tails)
    #     p_perm[i] = np.mean(larger, axis=0)

    # The line below accomplishes the same as above twice as fast, but could
    # run into memory errors if n_perm is greater than 1000
    p_perm = np.mean(tail_compare(diff, diff[:, np.newaxis]), axis=1)

    # Create binary clusters using the p value threshold
    b_act = tail_compare(1 - p_act, 1 - p_thresh, tails)
    b_perm = tail_compare(1 - p_perm, 1 - p_thresh, tails)

    # logger.info('Finding clusters')
    if ignore_adjacency is None:
        return time_cluster(b_act, b_perm, 1 - p_cluster)

    # If there are axes to ignore, we need to loop over them
    clusters = np.zeros(b_act.shape, dtype=int)
    for i in np.ndindex(tuple(sig1.shape[i] for i in ignore_adjacency)):
        index = tuple(j for j in i) + (slice(None),)
        clusters[index] = time_cluster(
            b_act[index], b_perm[(slice(None),) + index], 1 - p_cluster)

    return clusters


def _perm_iter(array: np.ndarray, perm: int, axis: int = 0, tails: int = 0
               ) -> np.ndarray:
    larger = tail_compare(array[perm],
                          array[np.arange(len(array)) != perm], tails)
    return np.mean(larger, axis=axis)


def make_data_shape(data_fix: np.ndarray, shape: tuple | list) -> np.ndarray:
    """Force the last dimension of data_fix to match the last dimension of
    shape.

    Takes the two arrays and checks if the last dimension of data_fix is
    smaller than the last dimension of shape. If there's more than two
    dimensions, it will rearrange the data to match the shape. If there's only
    two dimensions, it will repeat the signal to match the shape of data_like.
    If the last dimension of data_fix is larger than the last dimension of
    shape, it will return a subset of data_fix.

    Parameters
    ----------
    data_fix : array
        The data to reshape.
    shape : list | tuple
        The shape of data to match.

    Returns
    -------
    data_fix : array
        The reshaped data.
    """

    # Find the new shape
    x = 1
    for s in shape[1:]:
        x *= s
    trials = int(data_fix.size / x)
    temp = np.full((trials, *shape[1:]), np.nan)

    # Assign the data to the new shape, concatenating the first dimension along
    # the last dimension
    for i in np.ndindex(shape[1:-1]):
        index = (slice(None),) + tuple(j for j in i)
        temp[index].flat = data_fix[index].flat

    return temp


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
            return diff > obs_diff
        case 2:
            return np.abs(diff) > np.abs(obs_diff)
        case -1:
            return diff < obs_diff
        case _:
            raise ValueError('tails must be 1, 2, or -1')


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

    # Calculate the difference between the two groups averaged across
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


# @njit([t.Tuple((float64[:, ::1], complex128[:, ::1]))(
#     float64[:, :], complex128[:, :, ::1])], nogil=True, boundscheck=True,
#     fastmath=True, cache=True)
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
