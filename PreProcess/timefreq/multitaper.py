from typing import Union
from functools import singledispatch

import numpy as np
from mne.utils import logger, warn, verbose, fill_doc
from mne.epochs import BaseEpochs
from mne.io.base import BaseRaw
from mne.time_frequency import AverageTFR, tfr_multitaper
from mne import events_from_annotations, Epochs, event
from scipy import signal, fft

from PreProcess.timefreq.utils import crop_pad, to_samples
from PreProcess.timefreq.fastmath import rescale

ListNum = Union[int, float, np.ndarray, list, tuple]


def dpss_windows(N: int, half_nbw: float, Kmax: int, *, sym: bool = True,
                 norm: Union[int, str] = None, low_bias: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Compute Discrete Prolate Spheroidal Sequences.

    Will give of orders [0,Kmax-1] for a given frequency-spacing multiple
    NW and sequence length N.

    .. note:: Copied from NiTime.

    Parameters
    ----------
    N : int
        Sequence length.
    half_nbw : float
        Standardized half bandwidth corresponding to 2 * half_bw = BW*f0
        = BW*N/dt but with dt taken as 1.
    Kmax : int
        Number of DPSS windows to return is Kmax (orders 0 through Kmax-1).
    sym : bool
        Whether to generate a symmetric window (`True`, for filter design) or
        a periodic window (`False`, for spectral analysis). Default is
        `True`.
    norm : 2 | 'approximate' | 'subsample' | None
        Window normalization method. If ``'approximate'`` or ``'subsample'``,
        windows are normalized by the maximum, and a correction scale-factor
        for even-length windows is applied either using
        ``N**2/(N**2+half_nbw)`` ("approximate") or a FFT-based subsample shift
        ("subsample"). ``2`` uses the L2 norm. ``None`` (the default) uses
        ``"approximate"`` when ``Kmax=None`` and ``2`` otherwise.
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9. Default is ``True``.

    Returns
    -------
    v, e : tuple,
        The v array contains DPSS windows shaped (Kmax, N).
        e are the eigenvalues.

    Notes
    -----
    Tridiagonal form of DPSS calculation (Slepian, 1978)

    References
    ----------
    David S. Slepian. Prolate spheroidal wave functions, fourier analysis, and
    uncertainty-V: the discrete case. Bell System Technical Journal,
    57(5):1371–1430, 1978. doi:10.1002/j.1538-7305.1978.tb02104.x.
    """

    dpss, eigvals = signal.windows.dpss(
        N, half_nbw, Kmax, sym=sym, norm=norm, return_ratios=True)
    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            warn('Could not properly use low_bias, keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == N  # old nitime bug
    return dpss, eigvals


def spectra(x: np.ndarray, dpss: np.ndarray, sfreq: float,
            n_fft: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute significant tapered spectra.

    Parameters
    ----------
    x : array, shape=(..., n_times)
        Input signal
    dpss : array, shape=(n_tapers, n_times)
        The tapers
    sfreq : float
        The sampling frequency
    n_fft : int | None
        Length of the FFT. If None, the number of samples in the input signal
        will be used.

    Returns
    -------
    x_mt : array, shape=(..., n_tapers, n_times)
        The tapered spectra
    freqs : array
        The frequency points in Hz of the spectra
    """
    if n_fft is None:
        n_fft = x.shape[-1]  # round(sfreq * round(x.shape[-1]*pad_fact/sfreq))

    # remove mean (do not use in-place subtraction as it may modify input x)
    x = x - np.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = fft.rfftfreq(n_fft, 1. / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)),
                    dtype=np.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = fft.rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.)
    return x_mt, freqs

@fill_doc
@verbose
def params(n_times: int, sfreq: float, bandwidth: float,
           low_bias: bool = True, adaptive: bool = False,
           verbose: bool = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Triage windowing and multitaper parameters.

    Parameters
    ----------
    n_times : int
        The number of time points.
    sfreq : float
        The sampling frequency.
    bandwidth : float | None
        The bandwidth of the windows in Hz. If None, the half bandwidth will be
        4 Hz.
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
    %(verbose)s

    Returns
    -------
    window_fun : array, shape=(n_tapers, n_times)
        The window functions for each taper.
    eigenvals : array, shape=(n_tapers,)
        The eigenvalues for each taper.
    adaptive : bool
        Whether to use adaptive weights to combine the tapered spectra into PSD
        """
    # Compute standardized half-bandwidth
    if isinstance(bandwidth, str):
        logger.info('    Using standard spectrum estimation with "%s" window'
                    % (bandwidth,))
        window_fun = signal.get_window(bandwidth, n_times)[np.newaxis]
        return window_fun, np.ones(1), False

    if bandwidth is not None:
        half_nbw = float(bandwidth) * n_times / sfreq
    else:
        half_nbw = 4.
    if half_nbw < 0.5:
        raise ValueError(
            'bandwidth value %s yields a normalized bandwidth of %s < 0.5, '
            'use a value of at least %s'
            % (bandwidth, half_nbw, sfreq / n_times))

    # Compute DPSS windows
    n_tapers_max = int(np.floor(2 * half_nbw - 1))
    window_fun, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                       sym=True, low_bias=low_bias)
    logger.info('    Using multitaper spectrum estimation with %d DPSS '
                'windows' % len(eigvals))

    if adaptive and len(eigvals) < 3:
        warn('Not adaptively combining the spectral estimators due to a '
             'low number of tapers (%s < 3).' % (len(eigvals),))
        adaptive = False

    return window_fun, eigvals, adaptive


# def tfr(line, freqs, n_cycles, **kwargs):
#
#     if 'picks' in kwargs.keys():
#         line = line.copy().picks(kwargs.pop('picks'))
#
#     num_chans = len(line.info['ch_names'])
#
#     out = np.empty((num_chans, len(freqs), line._data.shape[2]))
#
#     # calculate time frequency response
#     for i, times in enumerate(n_cycles / freqs):
#         samps = to_samples(str(times) + "s", line.info['sfreq'])
#         window_fun, _, _ = params(samps, line.info['sfreq'], 10.0)
#         for j in range(num_chans):
#             x_p, freqs_new = spectra(line._data[:,j],
#             window_fun,line.info['sfreq'], pad_fact=1)
#             out[i, j, :] = x_p
#
#     power = AverageTFR(line.info, out, n_cycles / freqs, freqs,
#                         1, None, 'multitaper-power')


@fill_doc
@singledispatch
@verbose
def spectrogram(line: BaseEpochs, freqs: np.ndarray,
                baseline: BaseEpochs = None, n_cycles: np.ndarray = None,
                pad: str = "0s", correction: str = 'ratio',
                verbose: int = None, **kwargs) -> AverageTFR:
    """Calculate the multitapered, baseline corrected spectrogram

    Parameters
    ----------
    line : BaseEpochs
        The data to be processed
     %(freqs_tfr)s
    baseline : BaseEpochs
        The baseline to be used for correction
    %(n_cycles_tfr)s
    pad : str
        The amount of padding to be used in the spectrogram
    correction : str
        The type of baseline correction to be used
    %(verbose)s

    Notes
    -----
    %(time_bandwidth_tfr_notes)s
    %(temporal-window_tfr_notes)s

    Returns
    -------
    power : AverageTFR
        The multitapered, baseline corrected spectrogram
    """
    if n_cycles is None:
        n_cycles = freqs / 2

    power, itc = tfr_multitaper(line, freqs, n_cycles, verbose=verbose,
                                **kwargs)

    # crop the padding off the spectral estimates
    crop_pad(power, pad)

    if baseline is None:
        return power

    # apply baseline correction
    basepower, bitc = tfr_multitaper(baseline, freqs, n_cycles,
                                     verbose=verbose, **kwargs)
    crop_pad(basepower, pad)

    # set output data
    corrected_data = rescale(power._data, basepower._data, correction)

    return AverageTFR(power.info, corrected_data, power.times, freqs,
                      power.nave, power.comment, power.method)


@spectrogram.register
def _(line: BaseRaw, freqs: np.ndarray, line_event: str, tmin: float,
      tmax: float, base_event: str = None, base_tmin: float = None,
      base_tmax: float = None, n_cycles: np.ndarray = None, pad: str = "500ms",
      correction: str = 'ratio', **kwargs) -> AverageTFR:
    """Calculate the multitapered, baseline corrected spectrogram

    Parameters
    ----------
    line : BaseRaw
        The data to be processed
    freqs : array-like
        The frequencies to be used in the spectrogram
    line_event : str
        The event to be used for the spectrogram
    tmin : float
        The start time of the spectrogram
    tmax : float
        The end time of the spectrogram
    base_event : str
        The event to be used for the baseline
    base_tmin : float
        The start time of the baseline
    base_tmax : float
        The end time of the baseline
    n_cycles : array-like
        The number of cycles to be used in the spectrogram
    pad : str
        The amount of padding to be used in the spectrogram
    correction : str
        The type of baseline correction to be used

    Returns
    -------
    power : AverageTFR
        The multitapered, baseline corrected spectrogram
    """

    # determine the events
    events, ids = events_from_annotations(line)
    dat_ids = [ids[i] for i in event.match_event_names(ids, line_event)]

    # pad the data
    pad_secs = to_samples(pad, line.info['sfreq']) / line.info['sfreq']

    # Epoch the data
    data = Epochs(line, events, dat_ids, tmin - pad_secs,
                  tmax + pad_secs, baseline=None, preload=True)

    # run baseline corrected version
    if base_event is None:
        return spectrogram(data, freqs, None, n_cycles, pad, correction,
                           **kwargs)

    base_ids = [ids[i] for i in event.match_event_names(ids, base_event)]
    baseline = Epochs(line, events, base_ids, base_tmin - pad_secs,
                      base_tmax + pad_secs, baseline=None, preload=True)

    return spectrogram(data, freqs, baseline, n_cycles, pad, correction,
                       **kwargs)
