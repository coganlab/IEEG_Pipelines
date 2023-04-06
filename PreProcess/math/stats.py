import numpy as np
import scipy.stats as stats

from tqdm import tqdm
from numba import njit
from ..utils.utils import parallelize


def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, p_thresh: float,
                      p_cluster: float = None, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0) -> np.ndarray:
    """ Calculate significant clusters using permutation testing and cluster
    correction.

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
        The axis to perform the permutation test across.

    Returns
    -------
    clusters : array, shape (..., time)
        The binary array of significant clusters.
    """
    # check inputs
    if p_cluster is None:
        p_cluster = p_thresh
    if tails not in [1, 2, -1]:
        raise ValueError('tails must be 1, 2, or -1')
    if p_cluster > 1 or p_cluster < 0 or p_thresh > 1 or p_thresh < 0:
        raise ValueError('p_thresh and p_cluster must be between 0 and 1')

    # Make sure the data is the same shape
    sig2 = make_data_same(sig2, sig1)

    # Calculate the p value of difference between the two groups
    p_act, diff = time_perm_shuffle(sig1, sig2, n_perm, tails, axis, True)

    # Calculate the p value of the permutation distribution
    p_perm = np.zeros(diff.shape)
    for i in tqdm(range(diff.shape[0])):
        # p_perm is the probability of observing a difference as large as the
        # other permutations, or larger, by chance
        larger = tail_compare(diff[i], diff[np.arange(len(diff)) != i], tails)
        p_perm[i] = np.mean(larger, axis=0)
    p_all = np.concatenate((np.expand_dims(p_act, 0), p_perm), axis=0)

    # Create binary clusters using the z scores
    b_all = tail_compare(p_all, 1 - p_thresh, tails)
    b_act = b_all[0]
    b_perm = b_all[1:]

    # Find clusters
    clusters = np.zeros(b_act.shape, dtype=int)
    for i in tqdm(range(b_act.shape[0])):
        clusters_p = time_cluster(b_act[i], b_perm[:, i])
        clusters[i] = clusters_p > (1 - p_cluster)

    return clusters


def make_data_same(data_fix: np.ndarray, data_like: np.ndarray) -> np.ndarray:
    """Force the last dimension of data_fix to match the last dimension of
    data_like.

    Takes the two arrays and checks if the last dimension of data_fix is
    smaller than the last dimension of data_like. If there's more than two
    dimensions, it will rearrange the data to match the shape of data_like.
    If there's only two dimensions, it will repeat the signal to match the
    shape of data_like. If the last dimension of data_fix is larger than the
    last dimension of data_like, it will return a subset of data_fix.

    Parameters
    ----------
    data_fix : array
        The data to reshape.
    data_like : array
        The data to match the shape of.

    Returns
    -------
    data_fix : array
        The reshaped data.
    """
    if data_like.shape[-1] > data_fix.shape[-1]:
        if len(data_like.shape) > 2:
            trials = int(data_fix[:, 0].size / data_like.shape[-1])
            temp = np.full((trials, *data_like.shape[1:]), np.nan)
            for i in range(data_like.shape[1]):
                temp[:, i].flat = data_fix[:, i].flat
            data_fix = temp.copy()
        else:  # repeat the signal if it is only 2D
            data_fix = np.pad(data_fix, ((0, 0), (
                0, data_like.shape[-1] - data_fix.shape[-1])), 'reflect')
    elif data_like.shape[-1] < data_fix.shape[-1]:
        data_fix = data_fix[:, :data_like.shape[-1]]

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
    from skimage import measure
    act_clusters = measure.label(act, connectivity=1)
    perm_clusters = measure.label(perm, connectivity=1)

    # For each permutation in the passive data, determine the maximum cluster
    # size
    max_perm_cluster = np.zeros(perm_clusters.shape)
    for i in range(perm_clusters.shape[0]):
        for j in range(1, perm_clusters.max()+1):
            max_perm_cluster[i] = np.maximum(max_perm_cluster[i],
                                             perm_clusters[i] == j)

    # For each cluster in the active data, determine the proportion of
    # permutations that have a cluster of the same size or larger
    cluster_p_values = np.zeros(act_clusters.shape)
    for i in range(act_clusters.max()):
        # Get the cluster
        act_cluster = act_clusters == i+1
        # Get the cluster size
        act_cluster_size = np.sum(act_cluster)
        # Determine the proportion of permutations that have a cluster of the
        # same size or larger
        larger = tail_compare(act_cluster_size,
                              np.sum(max_perm_cluster, axis=1))
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
    if tails == 1:
        larger = diff > obs_diff
    elif tails == 2:
        larger = np.abs(diff) > np.abs(obs_diff)
    elif tails == -1:
        larger = diff < obs_diff
    else:
        raise ValueError('tails must be 1 or 2')

    return larger


def time_perm_shuffle(sig1: np.ndarray, sig2: np.ndarray, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0, return_perm: bool = False
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

    Returns
    -------
    p : np.ndarray, shape (..., time)
        The p-values for each time point.
        """
    # Concatenate the two signals for trial shuffling
    all_trial = np.concatenate((sig1, sig2), axis=axis)
    labels = np.concatenate((np.full(sig1.shape[axis], False, dtype=bool),
                             np.full(sig2.shape[axis], True, dtype=bool)))

    perm_labels = np.array(
        [np.random.permutation(labels) for _ in range(n_perm)])

    # Calculate the observed difference
    obs_diff = np.mean(sig1, axis) - np.mean(sig2, axis)

    # Calculate the difference between the two groups averaged across
    # trials at each time point
    fake_sig1 = np.array([np.take(all_trial, np.where(  # where False
        np.invert(perm_labels[i]))[0], axis=axis) for i in range(n_perm)])
    fake_sig2 = np.array([np.take(all_trial, np.where(  # where True
        perm_labels[i])[0], axis=axis) for i in range(n_perm)])
    diff = np.mean(fake_sig1, axis=axis + 1) - np.mean(fake_sig2,
                                                       axis=axis + 1)

    larger = tail_compare(diff, obs_diff, tails)

    # Calculate the p-value
    p = np.mean(larger, axis=0)

    if return_perm:
        return p, diff
    else:
        return p


def sum_squared(x: np.ndarray) -> np.ndarray:
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
