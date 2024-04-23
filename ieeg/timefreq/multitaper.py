from collections import Counter
from functools import cache, singledispatch
from typing import Union

import numpy as np
from mne import Epochs, event, events_from_annotations
from mne.epochs import BaseEpochs
from mne.io import Raw, base
from mne.time_frequency import AverageTFRArray, tfr_multitaper
from mne.utils import _pl, fill_doc, logger, verbose
from scipy import fft, signal, stats

from ieeg import ListNum
from ieeg.calc.scaling import rescale
from ieeg.calc.stats import sine_f_test
from ieeg.process import COLA, is_number
from ieeg.timefreq.utils import crop_pad, to_samples


class WindowingRemover(object):
    """Removes windowing artifacts from data.

    Parameters
    ----------
    sfreq : float
        The sampling frequency of the data.
    line_freqs : list of float
        The frequencies of the line noise.
    notch_width : list of float
        The notch widths for each line frequency.
    filter_length : int
        The length of the filter to use.
    low_bias : bool
        Whether to use a low bias filter.
    adaptive : bool
        Whether to use an adaptive filter.
    bandwidth : float
        The bandwidth of the multitaper windowing function.
    p_value : float
        The p-value to use in the F-test.
    verbose : bool
        Whether to print information.
    """

    @verbose
    def __init__(self, sfreq: float, line_freqs: ListNum,
                 notch_width: ListNum, filter_length: int, low_bias: bool,
                 adaptive: bool, bandwidth: float, p_value: float,
                 verbose: bool = None):
        self.sfreq = sfreq
        self.line_freqs = line_freqs
        self.notch_width = notch_width
        self.p_value = p_value
        self.filter_length = filter_length
        self.verbose = verbose
        self.low_bias = low_bias
        self.adaptive = adaptive
        self.bandwidth = bandwidth
        self.logger = logger
        self.rm_freqs = list()

    def dpss_windows(self, N: int, half_nbw: float, Kmax: int, *,
                     sym: bool = True, norm: Union[int, str] = None
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
            Whether to generate a symmetric window (`True`, for filter design)
            or a periodic window (`False`, for spectral analysis). Default is
            `True`.
        norm : 2 | 'approximate' | 'subsample' | None
            Window normalization method. If ``'approximate'`` or
            ``'subsample'``, windows are normalized by the maximum, and a
            correction scale-factor for even-length windows is applied either
            using ``N**2/(N**2+half_nbw)`` ("approximate") or a FFT-based
            subsample shift ("subsample"). ``2`` uses the L2 norm. ``None``
            (the default) uses ``"approximate"`` when ``Kmax=None`` and ``2``
            otherwise.

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
        David S. Slepian. Prolate spheroidal wave functions, fourier analysis,
        and uncertainty-V: the discrete case. Bell System Technical Journal,
        57(5):1371–1430, 1978. doi:10.1002/j.1538-7305.1978.tb02104.x.
        """

        dpss, eigvals = signal.windows.dpss(
            N, half_nbw, Kmax, sym=sym, norm=norm, return_ratios=True)
        if self.low_bias:
            idx = (eigvals > 0.9)
            if not idx.any():
                self.logger.warn('Could not properly use low_bias, keeping'
                                 'lowest-bias taper')
                idx = [np.argmax(eigvals)]
            dpss, eigvals = dpss[idx], eigvals[idx]
        assert len(dpss) > 0  # should never happen
        assert dpss.shape[1] == N  # old nitime bug
        return dpss, eigvals

    @fill_doc
    def params(self, n_times: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """Triage windowing and multitaper parameters.

        Parameters
        ----------
        n_times : int
            The number of time points.

        Returns
        -------
        window_fun : array, shape=(n_tapers, n_times)
            The window functions for each taper.
        eigenvals : array, shape=(n_tapers,)
            The eigenvalues for each taper.
        adaptive : bool
            Whether to use adaptive weights to combine the tapered spectra into
            PSD
            """
        # Compute standardized half-bandwidth
        if isinstance(self.bandwidth, str):
            self.logger.info(
                '    Using standard spectrum estimation with "%s" window'
                % (self.bandwidth,))
            window_fun = signal.get_window(self.bandwidth, n_times)[np.newaxis]
            return window_fun, np.ones(1), False

        if self.bandwidth is not None:
            half_nbw = float(self.bandwidth) * n_times / self.sfreq
        else:
            half_nbw = 4.
        if half_nbw < 0.5:
            raise ValueError(
                'bandwidth value %s yields a normalized bandwidth of %s < 0.5,'
                ' use a value of at least %s'
                % (self.bandwidth, half_nbw, self.sfreq / n_times))

        # Compute DPSS windows
        n_tapers_max = int(np.floor(2 * half_nbw - 1))
        window_fun, eigvals = self.dpss_windows(
            n_times, half_nbw, n_tapers_max, sym=True)
        self.logger.info('  Using multitaper spectrum estimation with %d DPSS '
                         'windows' % len(eigvals))

        if self.adaptive and len(eigvals) < 3:
            self.logger.warn('Not adaptively combining the spectral estimators'
                             ' due to a low number of tapers (%s < 3).' % (
                                 len(eigvals),))
            self.adaptive = False

        return window_fun, eigvals, self.adaptive

    @cache
    def get_thresh(self, n_times: int = None) -> tuple[np.ndarray, float]:
        """Get the window function and threshold for given time points.

        Parameters
        ----------
        n_times : int | None
            The number of time points. If None, the filter length will be used.

        Returns
        -------
        window_fun : array, shape=(n_tapers, n_times)
            The window functions for each taper
        threshold : float
            The threshold for the F-statistic.
        """

        if n_times is None:
            n_times = self.filter_length

        # figure out what tapers to use
        window_fun, _, _ = self.params(n_times)

        # F-stat of 1-p point
        threshold = stats.f.ppf(1 - self.p_value / n_times, 2,
                                2 * len(window_fun) - 2)
        return window_fun, threshold

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Remove line frequencies from data using multitaper method."""
        # Set default window function and threshold
        window_fun, thresh = self.get_thresh()
        n_times = x.shape[-1]
        n_samples = window_fun.shape[1]
        n_overlap = (n_samples + 1) // 2
        x_out = np.zeros_like(x)
        idx = [0]

        # Define how to process a chunk of data
        def process(x_):
            window_fun, thresh = self.get_thresh()
            out = _mt_remove(x_, self.sfreq, self.line_freqs, self.notch_width,
                             window_fun, thresh, self.get_thresh)
            self.rm_freqs.append(out[1])
            return (out[0],)  # must return a tuple

        # Define how to store a chunk of fully processed data (it's trivial)
        def store(x_):
            stop = idx[0] + x_.shape[-1]
            x_out[..., idx[0]:stop] += x_
            idx[0] = stop

        COLA(process, store, n_times, n_samples, n_overlap, self.sfreq,
             verbose=False).feed(x)
        assert idx[0] == n_times

        # report found frequencies, but do some sanitizing first by binning
        # into 1 Hz bins
        counts = Counter(sum((np.unique(np.round(ff)).tolist()
                              for f in self.rm_freqs for ff in f), list()))
        kind = 'Detected' if self.line_freqs is None else 'Removed'
        found_freqs = '\n'.join(f'    {freq:6.2f} : '
                                f'{counts[freq]:4d} window{_pl(counts[freq])}'
                                for freq in sorted(counts)) or '    None'
        self.logger.info(f'{kind} notch frequencies (Hz):\n{found_freqs}')

        x = x_out
        return x_out


def _mt_remove(x: np.ndarray, sfreq: float, line_freqs: ListNum,
               notch_widths: ListNum, window_fun: np.ndarray,
               threshold: float, get_thresh: callable,
               ) -> tuple[np.ndarray, list[float]]:
    """Use MT-spectrum to remove line frequencies.
    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """

    assert x.ndim == 1
    if x.shape[-1] != window_fun.shape[-1]:
        window_fun, threshold = get_thresh(x.shape[-1])
    # compute mt_spectrum (returning n_ch, n_tapers, n_freq)
    x_p, freqs = spectra(x[np.newaxis, :], window_fun, sfreq)
    f_stat, A = sine_f_test(window_fun, x_p)

    # find frequencies to remove
    indices = np.where(f_stat > threshold)[1]
    # pdf = 1-stats.f.cdf(f_stat, 2, window_fun.shape[0]-2)
    # indices = np.where(pdf < 1/x.shape[-1])[1]
    # specify frequencies within indicated ranges
    if line_freqs is not None and notch_widths is not None:
        if not isinstance(notch_widths, (list, tuple)) and is_number(
                notch_widths):
            notch_widths = [notch_widths] * len(line_freqs)
        ranges = [(freq - notch_width / 2, freq + notch_width / 2
                   ) for freq, notch_width in zip(line_freqs, notch_widths)]
        indices = [ind for ind in indices if any(
            lower <= freqs[ind] <= upper for (lower, upper) in ranges)]

    fits = list()
    # make "time" vector
    rads = 2 * np.pi * (np.arange(x.size) / float(sfreq))
    for ind in indices:
        c = 2 * A[0, ind]
        fit = np.abs(c) * np.cos(freqs[ind] * rads + np.angle(c))
        fits.append(fit)

    if len(fits) == 0:
        datafit = 0.0
    else:
        # fitted sinusoids are summed, and subtracted from data
        datafit = np.sum(fits, axis=0)

    return x - datafit, freqs[indices]


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
    x_mt = fft.rfft(x[:, np.newaxis, :] * dpss, n=n_fft, workers=1)
    # n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    # x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)),
    #                 dtype=np.complex128)
    # for idx, sig in enumerate(x):
    #     x_mt[idx] = fft.rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.)
    return x_mt, freqs


@fill_doc
@singledispatch
@verbose
def spectrogram(line: BaseEpochs, freqs: np.ndarray,
                baseline: BaseEpochs = None, n_cycles: np.ndarray = None,
                pad: str = "0s", correction: str = 'ratio',
                verbose: int = None, **kwargs) -> AverageTFRArray:
    """Calculate the multitapered, baseline corrected spectrogram

    Parameters
    ----------
    line : Epochs
        The data to be processed
    %(freqs_tfr)s
    baseline : Epochs
        The baseline to be used for correction
    %(n_cycles_tfr)s
    pad : str
        The amount of padding to be removed in the spectrogram
    correction : str
        The type of baseline correction to be used
    %(verbose)s

    Notes
    -----
    %(time_bandwidth_tfr_notes)s

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
    corrected_data = rescale(power._data, basepower._data, correction, axis=-1)

    return AverageTFRArray(power.info, corrected_data, power.times, freqs,
                           nave=power.nave, comment=power.comment,
                           method=power.method)


@spectrogram.register
def _(line: base.BaseRaw, freqs: np.ndarray, line_event: str, tmin: float,
      tmax: float, base_event: str = None, base_tmin: float = None,
      base_tmax: float = None, n_cycles: np.ndarray = None, pad: str = "500ms",
      correction: str = 'ratio', **kwargs) -> AverageTFRArray:
    """Calculate the multitapered, baseline corrected spectrogram

    Parameters
    ----------
    line : Raw
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
