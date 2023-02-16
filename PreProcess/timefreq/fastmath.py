from functools import singledispatch

import numpy as np
from mne.utils import logger, verbose
from mne import Epochs
from mne.time_frequency import EpochsSpectrum
from typing import Union


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
    """computes the F-statistic for sine wave in locally-white noise"""
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


def _log_rescale(baseline, mode='mean'):
    """Log the rescaling method."""
    if baseline is not None:
        msg = 'Applying baseline correction (mode: %s)' % mode
    else:
        msg = 'No baseline correction applied'
    return msg


@singledispatch
def rescale(data: np.ndarray, basedata: np.ndarray, mode: str = 'mean',
            copy: bool = True, verbose=None) -> np.ndarray:
    """Rescale (baseline correct) data.
    Parameters
    ----------
    data : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    basedata : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    %(baseline_rescale)s
    mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by
        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')
    copy : bool
        Whether to return a new instance or modify in place.
    picks : list of int | None
        Data to process along the axis=-2 (None, default, processes all).
    %(verbose)s
    Returns
    -------
    data_scaled: array
        Array of same shape as data after rescaling.
    """
    if copy:
        data = data.copy()
    if verbose is not False:
        msg = _log_rescale(basedata, mode)
        logger.info(msg)

    match mode:
        case 'mean':
            def fun(d, m, s):
                d -= m
        case 'ratio':
            def fun(d, m, s):
                d /= m
        case 'logratio':
            def fun(d, m, s):
                d /= m
                np.log10(d, out=d)
        case 'percent':
            def fun(d, m, s):
                d -= m
                d /= m
        case 'zscore':
            def fun(d, m, s):
                d -= m
                d /= s
        case 'zlogratio':
            def fun(d, m, s):
                d /= m
                np.log10(d, out=d)
                d /= s
        case _:
            raise NotImplementedError()

    mean = np.mean(basedata, axis=-1, keepdims=True)
    std = np.std(basedata, axis=-1, keepdims=True)
    fun(data, mean, std)
    return data


@rescale.register
def _(line,
      baseline,
      mode: str = 'mean', copy: bool = True, picks: list = 'data',
      verbose=None):
        basedata = baseline.pick(picks)._data
        if copy:
            line = line.copy()
        line.pick(picks)._data = rescale(line.pick(picks)._data, basedata,
                                         mode, False, verbose)
        return line