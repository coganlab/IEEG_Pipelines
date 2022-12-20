import operator
from collections import Counter
from functools import partial
from typing import TypeVar, Union

import numpy as np
from mne._ola import _COLA
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base, pick
from mne.parallel import parallel_func
from mne.utils import logger, _pl, warn, verbose
from scipy import stats, interpolate
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal.windows import dpss as sp_dpss

if __name__ in ['__main_'+'_', "PreProcess"]:
    from utils import ensure_int, validate_type, sum_squared
else:
    from .utils import ensure_int, validate_type, sum_squared

Signal = TypeVar("Signal", base.BaseRaw, BaseEpochs, Evoked)
ListNum = TypeVar("ListNum", int, float, np.ndarray, list, tuple)


@verbose
def line_filter(x: Signal, fs: float, freqs: ListNum = None,
                 filter_length: str = 'auto', notch_widths: ListNum = None,
                 mt_bandwidth: float = None, p_value: float = 0.05,
                 picks: ListNum = None, n_jobs: int = None,
                 copy: bool = True, *, verbose: Union[int, bool, str] = None) -> Signal:
    r"""Notch filter for the signal x.
    Applies a zero-phase notch filter to the signal x, operating on the last
    dimension.
    Parameters
    ----------
    x : array
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

    x = _check_filterable(x, 'notch filtered', 'notch_filter')
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

    xf = _mt_spectrum_proc(x, fs, freqs, notch_widths, mt_bandwidth,
                           p_value, picks, n_jobs, copy, filter_length)

    return xf


def _get_window_thresh(n_times, sfreq, mt_bandwidth, p_value):
    # max taper size chosen because it has an max error < 1e-3:
    # >>> np.max(np.diff(dpss_windows(953, 4, 100)[0]))
    # 0.00099972447657578449
    # so we use 1000 because it's the first "nice" number bigger than 953.
    # but if we have a new enough scipy,
    # it's only ~0.175 sec for 8 tapers even with 100000 samples
    dpss_n_times_max = 100000

    # figure out what tapers to use
    window_fun, _, _ = _compute_mt_params(
        n_times, sfreq, mt_bandwidth, False, False,
        interp_from=min(n_times, dpss_n_times_max), verbose=False)

    # F-stat of 1-p point
    threshold = stats.f.ppf(1 - p_value / n_times, 2, 2 * len(window_fun) - 2)
    return window_fun, threshold


def _mt_spectrum_proc(x, sfreq, line_freqs, notch_widths, mt_bandwidth,
                      p_value, picks, n_jobs, copy, filter_length):
    """Call _mt_spectrum_remove."""
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    if isinstance(filter_length, str) and filter_length == 'auto':
        filter_length = '10s'
    if filter_length is None:
        filter_length = x.shape[-1]
    filter_length = min(_to_samples(filter_length, sfreq, '', ''), x.shape[-1])
    get_wt = partial(
        _get_window_thresh, sfreq=sfreq, mt_bandwidth=mt_bandwidth,
        p_value=p_value)
    window_fun, threshold = get_wt(filter_length)
    parallel, p_fun, n_jobs = parallel_func(_mt_spectrum_remove_win, n_jobs)
    if n_jobs == 1:
        freq_list = list()
        for ii, x_ in enumerate(x):
            if ii in picks:
                x[ii], f = _mt_spectrum_remove_win(
                    x_, sfreq, line_freqs, notch_widths, window_fun, threshold,
                    get_wt)
                freq_list.append(f)
    else:
        data_new = parallel(p_fun(x_, sfreq, line_freqs, notch_widths,
                                  window_fun, threshold, get_wt)
                            for xi, x_ in enumerate(x)
                            if xi in picks)
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


def _mt_spectrum_remove_win(x, sfreq, line_freqs, notch_widths,
                            window_fun, threshold, get_thresh):
    n_times = x.shape[-1]
    n_samples = window_fun.shape[1]
    n_overlap = (n_samples + 1) // 2
    x_out = np.zeros_like(x)
    rm_freqs = list()
    idx = [0]

    # Define how to process a chunk of data
    def process(x_):
        out = _mt_spectrum_remove(
            x_, sfreq, line_freqs, notch_widths, window_fun, threshold,
            get_thresh)
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


def _mt_spectrum_remove(x, sfreq, line_freqs, notch_widths,
                        window_fun, threshold, get_thresh):
    """Use MT-spectrum to remove line frequencies.
    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """
    assert x.ndim == 1
    if x.shape[-1] != window_fun.shape[-1]:
        window_fun, threshold = get_thresh(x.shape[-1])
    # drop the even tapers
    n_tapers = len(window_fun)
    tapers_odd = np.arange(0, n_tapers, 2)
    tapers_even = np.arange(1, n_tapers, 2)
    tapers_use = window_fun[tapers_odd]

    # sum tapers for (used) odd prolates across time (n_tapers, 1)
    H0 = np.sum(tapers_use, axis=1)

    # sum of squares across tapers (1, )
    H0_sq = sum_squared(H0)

    # make "time" vector
    rads = 2 * np.pi * (np.arange(x.size) / float(sfreq))

    # compute mt_spectrum (returning n_ch, n_tapers, n_freq)
    x_p, freqs = _mt_spectra(x[np.newaxis, :], window_fun, sfreq)

    # sum of the product of x_p and H0 across tapers (1, n_freqs)
    x_p_H0 = np.sum(x_p[:, tapers_odd, :] *
                    H0[np.newaxis, :, np.newaxis], axis=1)

    # resulting calculated amplitudes for all freqs
    A = x_p_H0 / H0_sq

    if line_freqs is None:
        # figure out which freqs to remove using F stat

        # estimated coefficient
        x_hat = A * H0[:, np.newaxis]

        # numerator for F-statistic
        num = (n_tapers - 1) * (A * A.conj()).real * H0_sq
        # denominator for F-statistic
        den = (np.sum(np.abs(x_p[:, tapers_odd, :] - x_hat) ** 2, 1) +
               np.sum(np.abs(x_p[:, tapers_even, :]) ** 2, 1))
        den[den == 0] = np.inf
        f_stat = num / den

        # find frequencies to remove
        indices = np.where(f_stat > threshold)[1]
        rm_freqs = freqs[indices]
    else:
        # specify frequencies
        indices_1 = np.unique([np.argmin(np.abs(freqs - lf))
                               for lf in line_freqs])
        indices_2 = [np.logical_and(freqs > lf - nw / 2., freqs < lf + nw / 2.)
                     for lf, nw in zip(line_freqs, notch_widths)]
        indices_2 = np.where(np.any(np.array(indices_2), axis=0))[0]
        indices = np.unique(np.r_[indices_1, indices_2])
        rm_freqs = freqs[indices]

    fits = list()
    for ind in indices:
        c = 2 * A[0, ind]
        fit = np.abs(c) * np.cos(freqs[ind] * rads + np.angle(c))
        fits.append(fit)

    if len(fits) == 0:
        datafit = 0.0
    else:
        # fitted sinusoids are summed, and subtracted from data
        datafit = np.sum(fits, axis=0)

    return x - datafit, rm_freqs


def _mt_spectra(x, dpss, sfreq, n_fft=None):
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


@verbose
def _compute_mt_params(n_times, sfreq, bandwidth, low_bias, adaptive,
                       interp_from=None, verbose=None):
    """Triage windowing and multitaper parameters."""
    # Compute standardized half-bandwidth
    from scipy.signal import get_window
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
                                       low_bias=low_bias,
                                       interp_from=interp_from)
    logger.info('    Using multitaper spectrum estimation with %d DPSS '
                'windows' % len(eigvals))

    if adaptive and len(eigvals) < 3:
        warn('Not adaptively combining the spectral estimators due to a '
             'low number of tapers (%s < 3).' % (len(eigvals),))
        adaptive = False

    return window_fun, eigvals, adaptive


def dpss_windows(N, half_nbw, Kmax, low_bias=True, interp_from=None,
                 interp_kind='linear'):
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
    low_bias : bool
        Keep only tapers with eigenvalues > 0.9.
    interp_from : int (optional)
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.
        .. note:: If SciPy 1.1 or greater is available, interpolating
                  is likely not necessary as DPSS computations should be
                  sufficiently fast.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.
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

    # This np.int32 business works around a weird Windows bug, see
    # gh-5039 and https://github.com/scipy/scipy/pull/8608
    Kmax = np.int32(operator.index(Kmax))
    N = np.int32(operator.index(N))
    W = float(half_nbw) / N
    nidx = np.arange(N, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > N:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and N is: %s. ' % N
            e_s += 'Please enter interp_from smaller than N.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, half_nbw, Kmax, low_bias=False)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            tmp = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = tmp(np.linspace(0, this_d.shape[-1] - 1, N,
                                     endpoint=False))

            # Rescale:
            d_temp = d_temp / np.sqrt(sum_squared(d_temp))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        dpss = sp_dpss(N, half_nbw, Kmax)

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390

    # compute autocorr using FFT (same as nitime.utils.autocorr(dpss) * N)
    rxx_size = 2 * N - 1
    n_fft = next_fast_len(rxx_size)
    dpss_fft = rfft(dpss, n_fft)
    dpss_rxx = irfft(dpss_fft * dpss_fft.conj(), n_fft)
    dpss_rxx = dpss_rxx[:, :N]

    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            warn('Could not properly use low_bias, keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    assert dpss.shape[1] == N  # old nitime bug
    return dpss, eigvals


def next_fast_len(target):
    """Find the next fast size of input data to `fft`, for zero-padding, etc.
    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)
    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.
    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.
    Notes
    -----
    Copied from SciPy with minor modifications.
    """
    from bisect import bisect_left
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            p2 = 2 ** int(quotient - 1).bit_length()

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _prep_for_filtering(x, copy, picks=None):
    """Set up array as 2D for filtering ease."""
    x = _check_filterable(x)
    if copy is True:
        x = x.copy()
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


def _check_filterable(x, kind='filtered', alternative='filter'):
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


def _to_samples(filter_length, sfreq, phase, fir_design):
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
        if fir_design == 'firwin':
            filter_length += (filter_length - 1) % 2
    filter_length = ensure_int(filter_length, 'filter_length')
    return filter_length


if __name__ == "__main__":
    from preProcess import get_data
    import mne

    # %% Set up logging
    mne.set_log_file("output.log",
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    layout, raw, raw_dat, dat = get_data(53, "SentenceRep")
    filt = line_filter(raw.get_data(), raw.info['sfreq'], mt_bandwidth=5.0,
                       filter_length='20s', verbose='INFO')