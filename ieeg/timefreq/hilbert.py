"""Filterbank Hilbert transform — pure-Python orchestration + OpenMP inner kernel.

Replaces the previous Cython implementation in ``hilbert.pyx``. The
heavy lifting (FFT, gaussian filter bank, inverse FFT) was already
performed by ``scipy.fft`` and ``numpy``; the Cython annotations bought
nothing observable. The one piece that did genuinely benefit from
parallelism — building H and multiplying it by the per-channel Xf —
lives in ``_hilbert_kernel.c`` and is invoked from
:func:`extract_channel_wrapper` below.

Public API (unchanged from the Cython version):

- :func:`get_centers`
- :func:`filterbank_hilbert_first_half_wrapper`
- :func:`extract_channel_wrapper`

The old standalone OpenMP ``filterbank_hilbert`` cdef function had no
callers anywhere in the codebase and has been removed.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft

from ieeg.timefreq import _hilbert_kernel


# Default filter-bank constants — match the previous Cython defaults so
# downstream call sites (gamma.py) keep producing the same outputs.
_DEFAULT_OCT_SPACE = 1.0 / 7.0
_DEFAULT_F0 = 0.018
_DEFAULT_A1 = float(np.log10(0.39))
_DEFAULT_A2 = 0.5


def get_centers(Wn, octSpace: float = _DEFAULT_OCT_SPACE,
                f0: float = _DEFAULT_F0,
                a1: float = _DEFAULT_A1,
                a2: float = _DEFAULT_A2) -> np.ndarray:
    """Build the filter-bank centre frequencies.

    Parameters
    ----------
    Wn : tuple
        ``(minf, maxf)`` frequency range to use for the filter bank.
    octSpace : float
        Octave spacing once the spacing crosses 4 Hz.
    f0 : float
        Initial centre frequency.
    a1, a2 : float
        Coefficients controlling ``sigma_f = 10**(a1 + a2 * log10(cf))``.

    Returns
    -------
    cfs : np.ndarray, dtype float32
        Centre frequencies that fall inside ``[minf, maxf]``.
    """
    minf, maxf = Wn
    if minf >= maxf:
        raise ValueError(
            f"Upper bound of frequency range must be greater than lower "
            f"bound, but got lower bound of {minf} and upper bound of {maxf}"
        )
    maxfo = np.log2(maxf / f0)  # octave of max freq

    cfs = [f0]
    sigma_f = 10 ** (a1 + a2 * np.log10(f0))

    while np.log2(cfs[-1] / f0) < maxfo:
        if cfs[-1] < 4:
            cfs.append(cfs[-1] + sigma_f)
        else:
            # log spacing once we pass 4 Hz
            cfo = np.log2(cfs[-1] / f0) + octSpace
            cfs.append(f0 * (2 ** cfo))
        sigma_f = 10 ** (a1 + a2 * np.log10(cfs[-1]))

    cfs = np.array(cfs, dtype="float32")
    locator = np.logical_and(cfs >= minf, cfs <= maxf)
    if locator.sum() == 0:
        raise ValueError(
            "Frequency band is too narrow, so no filters in filterbank are "
            "placed inside. Try a wider frequency band."
        )
    return cfs[locator]


def filterbank_hilbert_first_half_wrapper(
    x: np.ndarray,
    fs: int,
    minf: float,
    maxf: float,
    oct_space: float = _DEFAULT_OCT_SPACE,
    f0: float = _DEFAULT_F0,
    a1: float = _DEFAULT_A1,
    a2: float = _DEFAULT_A2,
):
    """Prepare the per-channel inputs for the filterbank Hilbert.

    Computes the centre frequencies, the FFT of the input signal, the
    per-bin sigma values, and the Hilbert multiplier vector ``h``.
    Identical signature and return values to the previous Cython
    implementation.

    Returns
    -------
    (Xf, freqs, cfs, N, sds, h)
        - ``Xf`` complex64, shape (N, n_channels): FFT of ``x``.
        - ``freqs`` float32, shape (N // 2 + 1,): non-negative bin frequencies.
        - ``cfs`` float32, shape (n_filters,): filter-bank centre frequencies.
        - ``N`` int: FFT length (= ``x.shape[0]``).
        - ``sds`` float32, shape (n_filters,): per-filter gaussian sigma.
        - ``h`` complex64, shape (N, 1): Hilbert multiplier (reshaped).
    """
    cfs = get_centers((float(minf), float(maxf)), oct_space, f0, a1, a2)

    a = np.array([a1, a2], dtype="float32")
    # exponent[j] = a1 + a2 * log10(cfs[j])
    exponent = (
        np.concatenate(
            (
                np.ones((len(cfs), 1), dtype="float32"),
                np.log10(cfs)[:, np.newaxis],
            ),
            axis=1,
        )
        @ a
    )
    sigma_fs = np.power(10, exponent)
    sds = (sigma_fs * np.float32(np.sqrt(2))).astype("float32")

    N = x.shape[0]
    freqs = (np.arange(0, N // 2 + 1) * (fs * 1.0 / N)).astype("float32")
    Xf = fft(x, N, axis=0).astype("complex64")

    h = np.zeros(N, dtype="complex64")
    h[0] = 1
    h[1 : (N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1

    h_T = h[:, np.newaxis]
    return Xf, freqs, cfs, N, sds, h_T


def extract_channel_wrapper(
    Xf: np.ndarray,
    freqs: np.ndarray,
    cfs: np.ndarray,
    N: int,
    sds: np.ndarray,
    h: np.ndarray,
    minf,
    maxf,
) -> np.ndarray:
    """Compute the filterbank Hilbert envelope for one channel.

    Identical signature to the previous Cython implementation. The
    gaussian filter bank H and the ``Xf * H`` multiplication are fused
    into a single OpenMP-parallel C kernel
    (:func:`_hilbert_kernel.build_apply_H_c64`); the inverse FFT and
    amplitude extraction use scipy/numpy directly.
    """
    n_cfs = int(cfs.shape[0])

    # Coerce to the dtypes expected by the C kernel (no-op if already correct).
    freqs_c = np.ascontiguousarray(freqs, dtype="float32")
    cfs_c = np.ascontiguousarray(cfs, dtype="float32")
    sds_c = np.ascontiguousarray(sds, dtype="float32")
    Xf_c = np.ascontiguousarray(Xf, dtype="complex64")
    # h may come in as (N, 1) or (N,); kernel accepts either.
    h_c = np.ascontiguousarray(h, dtype="complex64")

    # Pre-allocate the (N, n_cfs) complex64 buffer for Xf * H.
    Xf_H = np.empty((int(N), n_cfs), dtype="complex64")

    _hilbert_kernel.build_apply_H_c64(
        freqs_c, cfs_c, sds_c, h_c, Xf_c, Xf_H,
    )

    hilb_channel = ifft(Xf_H, N, axis=0).astype("complex64")
    band_locator = np.logical_and(cfs >= minf, cfs <= maxf)
    hilb_amp = np.abs(hilb_channel[:, band_locator]).astype("float32")
    return hilb_amp
