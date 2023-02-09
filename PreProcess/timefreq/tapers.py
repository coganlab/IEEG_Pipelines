import numpy as np
from mne.utils import logger, _pl, warn, verbose
from scipy import signal, fft
from numpy.typing import ArrayLike
from typing import Union, List, Tuple, Dict, Any, TypeVar

from .fastmath import sine_f_test
from PreProcess.utils.utils import is_number

ListNum = TypeVar("ListNum", int, float, np.ndarray, list, tuple)


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


def mt_spectra(x: ArrayLike, dpss: ArrayLike, sfreq: float,
                n_fft: int = None) -> (ArrayLike, ArrayLike):
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
        n_fft = x.shape[-1]

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


@verbose
def mt_params(n_times, sfreq: float, bandwidth: float, low_bias: bool,
                       adaptive: bool,
                       verbose: bool = None):
    """Triage windowing and multitaper parameters."""
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