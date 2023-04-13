import numpy as np

from skimage import measure
from tqdm import tqdm
from mne.utils import logger


def mean_diff(group1: np.ndarray, group2: np.ndarray,
              axis: int | tuple[int] = None) -> np.ndarray | float:
    """ Calculate the mean difference between two groups.

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

    avg1 = np.mean(group1, axis=axis)
    avg2 = np.mean(group2, axis=axis)

    return avg1 - avg2


def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, p_thresh: float,
                      p_cluster: float = None, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0,
                      stat_func: callable = mean_diff) -> np.ndarray:
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

    # Make sure the data is the same shape
    sig2 = make_data_shape(sig2, sig1.shape)

    # Calculate the p value of difference between the two groups
    logger.info('Permuting events in shuffle test')
    p_act, diff = time_perm_shuffle(sig1, sig2, n_perm, tails, axis, True,
                                    stat_func)

    # Calculate the p value of the permutation distribution
    logger.info('Calculating permutation distribution')
    p_perm = np.zeros(diff.shape)
    for i in tqdm(range(diff.shape[0]), 'Permutations'):
        # p_perm is the probability of observing a difference as large as the
        # other permutations, or larger, by chance
        larger = tail_compare(diff[i], diff[np.arange(len(diff)) != i], tails)
        p_perm[i] = np.mean(larger, axis=0)

    # The line below accomplishes the same as above twice as fast, but could
    # run into memory errors if n_perm is greater than 1000
    # p_perm = np.mean(tail_compare(diff, diff[:, np.newaxis]), axis=1)

    # Create binary clusters using the p value threshold
    b_act = tail_compare(1 - p_act, 1 - p_thresh, tails)
    b_perm = tail_compare(1 - p_perm, 1 - p_thresh, tails)

    logger.info('Finding clusters')
    clusters = np.zeros(b_act.shape, dtype=int)
    for i in tqdm(range(b_act.shape[0]), 'Channels'):
        clusters[i] = time_cluster(b_act[i], b_perm[:, i], 1 - p_cluster)

    return clusters


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
    if shape[-1] > data_fix.shape[-1]:
        if len(shape) > 2:
            trials = int(data_fix[:, 0].size / shape[-1])
            temp = np.full((trials, *shape[1:]), np.nan)
            for i in range(shape[1]):
                temp[:, i].flat = data_fix[:, i].flat
            data_fix = temp.copy()
        else:  # repeat the signal if it is only 2D
            data_fix = np.pad(data_fix, ((0, 0), (
                0, shape[-1] - data_fix.shape[-1])), 'reflect')
    elif shape[-1] < data_fix.shape[-1]:
        data_fix = data_fix[:, :shape[-1]]

    return data_fix


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
            larger = diff > obs_diff
        case 2:
            larger = np.abs(diff) > np.abs(obs_diff)
        case -1:
            larger = diff < obs_diff
        case _:
            raise ValueError('tails must be 1, 2, or -1')

    return larger


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
    for i in tqdm(range(n_perm), "Permuting resamples"):
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
