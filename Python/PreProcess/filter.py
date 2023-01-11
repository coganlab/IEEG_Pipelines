from collections import Counter
from functools import partial
from typing import TypeVar, Union, List

import numpy as np
from numpy.typing import ArrayLike
from mne._ola import _COLA
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base, pick
from mne.utils import logger, _pl, warn, verbose
from scipy import stats, interpolate
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import dpss as sp_dpss
from scipy.signal import get_window

if __name__ in ['__main_' + '_', "PreProcess"]:
    from utils import ensure_int, validate_type, parallelize, is_number
    from fastmath import sine_f_test
else:
    from .utils import ensure_int, validate_type, parallelize, is_number
    from .fastmath import sine_f_test

Signal = TypeVar("Signal", base.BaseRaw, BaseEpochs, Evoked)
ListNum = TypeVar("ListNum", int, float, np.ndarray, list, tuple)


@verbose
def line_filter(raw: Signal, fs: float = None, freqs: ListNum = None,
                filter_length: str = 'auto', notch_widths: ListNum = None,
                mt_bandwidth: float = None, p_value: float = 0.05,
                picks: ListNum = None, n_jobs: int = None,
                copy: bool = True, *, verbose: Union[int, bool, str] = None) -> Signal:
    r"""Notch filter for the signal x.
    Applies a zero-phase notch filter to the signal x, operating on the last
    dimension.
    Parameters
    ----------
    raw : array
        Signal to filter.
    fs : float
        Sampling rate in Hz.
    freqs : float | array of float | None
        Frequencies to notch filter in Hz, e.g. np.arange(60, 241, 60).
        None can only be used with the mode 'spectrum_fit', where an F
        test is used to find sinusoidal components.
    %(filter_length_notch)s
    notch_widths : float | array of float | None
        Width of the stop band (centred at each freq in freqs) in Hz.
        If None, freqs / 200 is used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'spectrum_fit' mode.
    p_value : float
        P-value to use in F-test thresholding to determine significant
        sinusoidal components to remove when method='spectrum_fit' and
        freqs=None. Note that this will be Bonferroni corrected for the
        number of frequencies, so large p-values may be justified.
    %(picks_nostr)s
        Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    %(n_jobs_fir)s
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    %(phase)s
    %(verbose)s
    Returns
    -------
    xf : array
        The x array filtered.
    See Also
    --------
    filter_data
    resample
    Notes
    -----
    The frequency response is (approximately) given by::
        1-|----------         -----------
          |          \       /
      |H| |           \     /
          |            \   /
          |             \ /
        0-|              -
          |         |    |    |         |
          0        Fp1 freq  Fp2       Nyq
    For each freq in freqs, where ``Fp1 = freq - trans_bandwidth / 2`` and
    ``Fs2 = freq + trans_bandwidth / 2``.
    References
    ----------
    Multi-taper removal is inspired by code from the Chronux toolbox, see
    www.chronux.org and the book "Observed Brain Dynamics" by Partha Mitra
    & Hemant Bokil, Oxford University Press, New York, 2008. Please
    cite this in publications if method 'spectrum_fit' is used.
    """
    if fs is None:
        fs = raw.info["sfreq"]
    if copy:
        filt = raw.copy()
    else:
        filt = raw
    x = _check_filterable(filt.get_data("data"), 'notch filtered', 'notch_filter')
    if freqs is not None:
        freqs = np.atleast_1d(freqs)
        # Only have to deal with notch_widths for non-autodetect
        if notch_widths is None:
            notch_widths = freqs / 200.0
        elif np.any(notch_widths < 0):
            raise ValueError('notch_widths must be >= 0')
        else:
            notch_widths = np.atleast_1d(notch_widths)
            if len(notch_widths) == 1:
                notch_widths = notch_widths[0] * np.ones_like(freqs)
            elif len(notch_widths) != len(freqs):
                raise ValueError('notch_widths must be None, scalar, or the '
                                 'same length as freqs')

    data_idx = [ch_t in set(raw.get_channel_types(only_data_chs=True)
                            ) for ch_t in raw.get_channel_types()]
    filt._data[data_idx] = mt_spectrum_proc(x, fs, freqs, notch_widths, mt_bandwidth,
                                             p_value, picks, n_jobs, filter_length)

    return filt


def mt_spectrum_proc(x: ArrayLike, sfreq: float, line_freqs: ListNum,
                     notch_widths: ListNum, mt_bandwidth: float,
                     p_value: float, picks: list, n_jobs: int,
                     filter_length: Union[str, int]) -> ArrayLike:
    """Call _mt_spectrum_remove."""
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, picks)

    # convert filter length to samples
    if isinstance(filter_length, str) and filter_length == 'auto':
        filter_length = '10s'
    if filter_length is None:
        filter_length = x.shape[-1]
    filter_length = min(_to_samples(filter_length, sfreq), x.shape[-1])

    # Define adaptive windowing function
    get_wt = partial(
        _get_window_thresh, sfreq=sfreq, bandwidth=mt_bandwidth,
        p_value=p_value)

    # Set default window function and threshold
    window_fun, threshold = get_wt(filter_length)

    # Execute channel wise sine wave detection and filtering
    if n_jobs == 1:
        freq_list = list()
        for ii, x_ in enumerate(x):
            if ii in picks:
                x[ii], f = _mt_spectrum_remove_win(
                    x_, sfreq, line_freqs, notch_widths, window_fun, threshold,
                    get_wt)
                freq_list.append(f)
    else:
        runs = [x_ for xi, x_ in enumerate(x) if xi in picks]
        data_new = parallelize(_mt_spectrum_remove_win, runs, n_jobs,
                               sfreq, line_freqs, notch_widths, window_fun,
                               threshold, get_wt)
        freq_list = [d[1] for d in data_new]
        data_new = np.array([d[0] for d in data_new])
        x[picks, :] = data_new

    # report found frequencies, but do some sanitizing first by binning into
    # 1 Hz bins
    counts = Counter(sum((np.unique(np.round(ff)).tolist()
                          for f in freq_list for ff in f), list()))
    kind = 'Detected' if line_freqs is None else 'Removed'
    found_freqs = '\n'.join(f'    {freq:6.2f} : '
                            f'{counts[freq]:4d} window{_pl(counts[freq])}'
                            for freq in sorted(counts)) or '    None'
    logger.info(f'{kind} notch frequencies (Hz):\n{found_freqs}')

    x.shape = orig_shape
    return x


def _mt_spectrum_remove_win(x: np.ndarray, sfreq: float, line_freqs: ListNum,
                            notch_widths: ListNum, window_fun: np.ndarray,
                            threshold: float, get_thresh: partial
                            ) -> (ArrayLike, List[float]):
    n_times = x.shape[-1]
    n_samples = window_fun.shape[1]
    n_overlap = (n_samples + 1) // 2
    x_out = np.zeros_like(x)
    rm_freqs = list()
    idx = [0]

    # Define how to process a chunk of data
    def process(x_):
        out = _mt_spectrum_remove(
            x_, sfreq, line_freqs, notch_widths, window_fun, threshold, get_thresh)
        rm_freqs.append(out[1])
        return (out[0],)  # must return a tuple

    # Define how to store a chunk of fully processed data (it's trivial)
    def store(x_):
        stop = idx[0] + x_.shape[-1]
        x_out[..., idx[0]:stop] += x_
        idx[0] = stop

    _COLA(process, store, n_times, n_samples, n_overlap, sfreq,
          verbose=False).feed(x)
    assert idx[0] == n_times
    return x_out, rm_freqs


def _mt_spectrum_remove(x: np.ndarray, sfreq: float, line_freqs: ListNum, notch_widths: ListNum,
                        window_fun: np.ndarray, threshold: float, get_thresh: partial) -> (ArrayLike, List[float]):
    """Use MT-spectrum to remove line frequencies.
    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """

    assert x.ndim == 1
    if x.shape[-1] != window_fun.shape[-1]:
        window_fun, threshold = get_thresh(x.shape[-1])
    # compute mt_spectrum (returning n_ch, n_tapers, n_freq)
    x_p, freqs = _mt_spectra(x[np.newaxis, :], window_fun, sfreq)
    f_stat, A = sine_f_test(window_fun, x_p)

    # find frequencies to remove
    indices = np.where(f_stat > threshold)[1]

    # specify frequencies
    if line_freqs is not None and notch_widths is not None:
        if not isinstance(notch_widths, (list, tuple)) and is_number(notch_widths):
            notch_widths = [notch_widths] * len(line_freqs)
        ranges = [(freq - notch_width/2, freq + notch_width/2
                   ) for freq, notch_width in zip(line_freqs, notch_widths)]
        indices = [ind for ind in indices if any(
            lower <= freqs[ind] <= upper for (lower, upper) in ranges)]
    # indices_1 = np.unique([np.argmin(np.abs(freqs - lf))
    #                        for lf in line_freqs])
    # indices_2 = [np.logical_and(freqs > lf - nw / 2., freqs < lf + nw / 2.)
    #              for lf, nw in zip(line_freqs, notch_widths)]
    # indices_2 = np.where(np.any(np.array(indices_2), axis=0))[0]
    # indices = np.unique(np.r_[indices_1, indices_2])

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


def _mt_spectra(x: ArrayLike, dpss: ArrayLike, sfreq: float,
                n_fft: int = None) -> (ArrayLike, ArrayLike):
    """Compute tapered spectra.
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
        n_fft = x.shape[-1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    x = x - np.mean(x, axis=-1, keepdims=True)

    # only keep positive frequencies
    freqs = rfftfreq(n_fft, 1. / sfreq)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, len(freqs)),
                    dtype=np.complex128)
    for idx, sig in enumerate(x):
        x_mt[idx] = rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)
    # Adjust DC and maybe Nyquist, depending on one-sided transform
    x_mt[..., 0] /= np.sqrt(2.)
    if n_fft % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.)
    return x_mt, freqs


def dpss_windows(N, half_nbw, Kmax, *, sym=True, norm=None, low_bias=True,
                 interp_from=None, interp_kind=None):
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
        Whether to generate a symmetric window (``True``, for filter design) or
        a periodic window (``False``, for spectral analysis). Default is
        ``True``.
        .. versionadded:: 1.3
    norm : 2 | ``'approximate'`` | ``'subsample'`` | None
        Window normalization method. If ``'approximate'`` or ``'subsample'``,
        windows are normalized by the maximum, and a correction scale-factor
        for even-length windows is applied either using
        ``N**2/(N**2+half_nbw)`` ("approximate") or a FFT-based subsample shift
        ("subsample"). ``2`` uses the L2 norm. ``None`` (the default) uses
        ``"approximate"`` when ``Kmax=None`` and ``2`` otherwise.
        .. versionadded:: 1.3
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9.
    interp_from : int | None
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.
        .. deprecated:: 1.3
           The ``interp_from`` option is deprecated and will be
           removed in version 1.4. Modern implementations can handle large
           values of ``N`` so interpolation is no longer necessary; any value
           passed here will be ignored.
    interp_kind : str | None
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.
        .. deprecated:: 1.3
           The ``interp_kind`` option is deprecated and will be
           removed in version 1.4. Modern implementations can handle large
           values of ``N`` so interpolation is no longer necessary; any value
           passed here will be ignored.
    Returns
    -------
    v, e : tuple,
        The v array contains DPSS windows shaped (Kmax, N).
        e are the eigenvalues.
    Notes
    -----
    Tridiagonal form of DPSS calculation from :footcite:`Slepian1978`.
    References
    ----------
    .. footbibliography::
    """

    if interp_from is not None:
        warn('The ``interp_from`` option is deprecated and will be removed in '
             'version 1.4.', FutureWarning)
    if interp_kind is not None:
        warn('The ``interp_kind`` option is deprecated and will be removed in '
             'version 1.4.', FutureWarning)

    dpss, eigvals = sp_dpss(N, half_nbw, Kmax, sym=sym, norm=norm,
                            return_ratios=True)
    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            warn('Could not properly use low_bias, keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == N  # old nitime bug
    return dpss, eigvals


def _get_window_thresh(n_times, sfreq, bandwidth, p_value):
    # figure out what tapers to use
    window_fun, _, _ = _compute_mt_params(
        n_times, sfreq, bandwidth, False, False, verbose=False)

    # F-stat of 1-p point
    threshold = stats.f.ppf(1 - p_value / n_times, 2, 2 * len(window_fun) - 2)
    return window_fun, threshold


def _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive,
                       verbose=None):
    """Triage windowing and multitaper parameters."""
    # Compute standardized half-bandwidth
    if isinstance(bandwidth, str):
        logger.info('    Using standard spectrum estimation with "%s" window'
                    % (bandwidth,))
        window_fun = get_window(bandwidth, n_times)[np.newaxis]
        return window_fun, np.ones(1), False

    if bandwidth is not None:
        half_nbw = float(bandwidth) * n_times / (2. * sfreq)
    else:
        half_nbw = 4.
    if half_nbw < 0.5:
        raise ValueError(
            'bandwidth value %s yields a normalized bandwidth of %s < 0.5, '
            'use a value of at least %s'
            % (bandwidth, half_nbw, sfreq / n_times))

    # Compute DPSS windows
    n_tapers_max = int(2 * half_nbw)
    window_fun, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                       sym=False, low_bias=low_bias)
    logger.info('    Using multitaper spectrum estimation with %d DPSS '
                'windows' % len(eigvals))

    if adaptive and len(eigvals) < 3:
        warn('Not adaptively combining the spectral estimators due to a '
             'low number of tapers (%s < 3).' % (len(eigvals),))
        adaptive = False

    return window_fun, eigvals, adaptive


def _prep_for_filtering(x: ArrayLike, picks: list = None) -> ArrayLike:
    """Set up array as 2D for filtering ease."""
    x = _check_filterable(x)
    orig_shape = x.shape
    x = np.atleast_2d(x)
    picks = pick._picks_to_idx(x.shape[-2], picks)
    x.shape = (np.prod(x.shape[:-1]), x.shape[-1])
    if len(orig_shape) == 3:
        n_epochs, n_channels, n_times = orig_shape
        offset = np.repeat(np.arange(0, n_channels * n_epochs, n_channels),
                           len(picks))
        picks = np.tile(picks, n_epochs) + offset
    elif len(orig_shape) > 3:
        raise ValueError('picks argument is not supported for data with more'
                         ' than three dimensions')
    assert all(0 <= pick < x.shape[0] for pick in picks)  # guaranteed by above

    return x, orig_shape, picks


def _check_filterable(x: Union[Signal, ArrayLike], kind: str = 'filtered', alternative: str = 'filter') -> np.ndarray:
    # Let's be fairly strict about this -- users can easily coerce to ndarray
    # at their end, and we already should do it internally any time we are
    # using these low-level functions. At the same time, let's
    # help people who might accidentally use low-level functions that they
    # shouldn't use by pushing them in the right direction
    if isinstance(x, (base.BaseRaw, BaseEpochs, Evoked)):
        try:
            name = x.__class__.__name__
        except Exception:
            pass
        else:
            raise TypeError(
                'This low-level function only operates on np.ndarray '
                f'instances. To get a {kind} {name} instance, use a method '
                f'like `inst_new = inst.copy().{alternative}(...)` '
                'instead.')
    validate_type(x, (np.ndarray, list, tuple))
    x = np.asanyarray(x)
    if x.dtype != np.float64:
        raise ValueError('Data to be %s must be real floating, got %s'
                         % (kind, x.dtype,))
    return x


def _to_samples(filter_length, sfreq):
    validate_type(filter_length, (str, int))
    if isinstance(filter_length, str):
        filter_length = filter_length.lower()
        err_msg = ('filter_length, if a string, must be a '
                   'human-readable time, e.g. "10s", or "auto", not '
                   '"%s"' % filter_length)
        if filter_length.lower().endswith('ms'):
            mult_fact = 1e-3
            filter_length = filter_length[:-2]
        elif filter_length[-1].lower() == 's':
            mult_fact = 1
            filter_length = filter_length[:-1]
        else:
            raise ValueError(err_msg)
        # now get the number
        try:
            filter_length = float(filter_length)
        except ValueError:
            raise ValueError(err_msg)
        filter_length = max(int(np.ceil(filter_length * mult_fact *
                                        sfreq)), 1)
    filter_length = ensure_int(filter_length, 'filter_length')
    return filter_length


if __name__ == "__main__":
    from preProcess import get_data, open_dat_file
    import mne

    # %% Set up logging
    mne.set_log_file("output.log",
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    layout, raw, D_dat_raw, D_dat_filt = get_data(53, "SentenceRep")
    filt = line_filter(raw, mt_bandwidth=5.0, n_jobs=None,
                       filter_length='20s', verbose=10,
                       freqs=[60, 120, 180, 240], notch_widths=20)
    raw_dat = open_dat_file(D_dat_raw, raw.copy().ch_names)
    dat = open_dat_file(D_dat_filt, raw.copy().ch_names)
