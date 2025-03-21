from functools import singledispatch

import numpy as np
from mne import Epochs
from mne.epochs import BaseEpochs
from mne.time_frequency import AverageTFR, EpochsTFR
from mne.utils import logger, verbose
from ieeg.calc.stats import dist


def _log_rescale(baseline, mode='mean'):
    """Log the rescaling method."""
    if baseline is not None:
        msg = 'Applying baseline correction (mode: %s)' % mode
    else:
        msg = 'No baseline correction applied'
    return msg


@singledispatch
def rescale(data: np.ndarray, basedata: np.ndarray, mode: str = 'mean',
            copy: bool = False, axis: tuple[int] | int = -1) -> np.ndarray:
    """Rescale (baseline correct) data.

    Implement a variety of baseline correction methods. The data is
    modified in place by default.

    Parameters
    ----------
    data : array | mne.Epochs | mne.EpochsTFR
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    basedata : array
        It can be of any shape. The last dimension should be time, and the
        other dimensions should be the same as data.
    mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio',\
        default 'mean', optional
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
        dividing by the standard deviation of log baseline values ('zlogratio')
    copy : bool, optional
        Whether to return a new instance or modify in place.
    axis : int or tuple[int], optional


    Returns
    -------
    data_scaled: array
        Array of same shape as data after rescaling.
    """
    if copy:
        data = data.copy()

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
    mean, std = dist(basedata, axis=axis, mode='std', ddof=1, keepdims=True)
    fun(data, mean, std)
    return data


@rescale.register
@verbose
def _(line: BaseEpochs, baseline: BaseEpochs,
      mode: str = 'mean', copy: bool = False, picks: list = 'data',
      verbose=None) -> Epochs:
    """Rescale (baseline correct) Epochs"""
    if copy:
        line: Epochs = line.copy()
    if verbose is not False:
        msg = _log_rescale(baseline, mode)
        logger.info(msg)

    # Average the baseline across epochs
    basedata = baseline.pick(picks)._data
    axes = list(range(basedata.ndim))

    # within channels
    axes.pop(1)

    # If time frequency then within frequency
    if isinstance(line, EpochsTFR):
        axes = (0, 3)
    elif isinstance(line, AverageTFR):
        axes = 2
    else:
        axes = tuple(axes)

    line.pick(picks)._data = rescale(line.pick(picks)._data, basedata, mode,
                                     False, axes)
    return line


@rescale.register
@verbose
def _(line: EpochsTFR, baseline: EpochsTFR,
      mode: str = 'mean', copy: bool = False, picks: list = 'data',
      verbose=None) -> Epochs:
    """Rescale (baseline correct) Epochs"""
    if copy:
        line: Epochs = line.copy()
    if verbose is not False:
        msg = _log_rescale(baseline, mode)
        logger.info(msg)

    # Average the baseline across epochs
    basedata = baseline.pick(picks)._data

    # If time frequency then within frequency
    axes = (0, 3)

    line.pick(picks)._data = rescale(line.pick(picks)._data, basedata, mode,
                                     False, axes)
    return line


@rescale.register
@verbose
def _(line: AverageTFR, baseline: AverageTFR,
      mode: str = 'mean', copy: bool = False, picks: list = 'data',
      verbose=None) -> Epochs:
    """Rescale (baseline correct) Epochs"""
    if copy:
        line: Epochs = line.copy()
    if verbose is not False:
        msg = _log_rescale(baseline, mode)
        logger.info(msg)

    # Average the baseline across epochs
    basedata = baseline.pick(picks)._data

    # If time frequency then within frequency
    axes = 2

    line.pick(picks)._data = rescale(line.pick(picks)._data, basedata, mode,
                                     False, axes)
    return line
