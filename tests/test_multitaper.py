"""Coverage for ``ieeg.timefreq.multitaper``.

Targets the uncovered branches in ``WindowingRemover``:
- string bandwidth (line 145-149)
- numeric bandwidth path (line 151-152)
- None bandwidth (line 153-154)
- bandwidth too low (line 155-159)
- low_bias warning path (line 115-118)
- adaptive-with-few-tapers warning (line 168-172)

Plus standalone tests for ``spectra`` and the scalar-notch_widths
branch in ``_mt_remove`` (line 268-270).
"""
from __future__ import annotations

import numpy as np
import pytest

from ieeg.timefreq.multitaper import WindowingRemover, spectra, _mt_remove


# -------------------------------------------------------------------- #
# WindowingRemover.dpss_windows
# -------------------------------------------------------------------- #

def _wr(**overrides):
    """Build a minimal WindowingRemover with sensible defaults."""
    defaults = dict(
        sfreq=1000.0,
        line_freqs=[60.0],
        notch_width=[2.0],
        filter_length=512,
        low_bias=True,
        adaptive=False,
        bandwidth=5.0,
        p_value=0.05,
        verbose=False,
    )
    defaults.update(overrides)
    return WindowingRemover(**defaults)


def test_dpss_with_low_bias_keeps_high_eigval_tapers():
    """Most tapers should pass the eigval>0.9 filter; result is non-empty."""
    wr = _wr(low_bias=True)
    dpss, eig = wr.dpss_windows(N=256, half_nbw=4.0, Kmax=7)
    assert dpss.shape[1] == 256
    assert dpss.shape[0] >= 1
    assert (eig > 0.9).all()


def test_dpss_without_low_bias_returns_all_tapers():
    wr = _wr(low_bias=False)
    dpss, eig = wr.dpss_windows(N=256, half_nbw=4.0, Kmax=7)
    # All Kmax requested tapers are kept.
    assert dpss.shape[0] == 7


def test_dpss_low_bias_fallback_when_no_eig_above_threshold():
    """With a tiny half_nbw, no eigvals reach 0.9 and the warn-path
    fires (line 115-118), keeping the single highest-bias taper."""
    wr = _wr(low_bias=True)
    # Very small half_nbw → all eigvals are low.
    dpss, eig = wr.dpss_windows(N=128, half_nbw=0.6, Kmax=1)
    assert dpss.shape[0] == 1


# -------------------------------------------------------------------- #
# WindowingRemover.params
# -------------------------------------------------------------------- #

def test_params_string_bandwidth_uses_scipy_window():
    """String bandwidth → standard spectrum estimation (line 144-149)."""
    wr = _wr(bandwidth="hann")
    window_fun, eig, adaptive = wr.params(n_times=512)
    assert window_fun.shape == (1, 512)
    assert eig.shape == (1,)
    assert eig[0] == 1.0
    assert adaptive is False


def test_params_numeric_bandwidth_computes_half_nbw():
    """Numeric bandwidth path (line 151-152)."""
    wr = _wr(bandwidth=4.0)
    window_fun, eig, adaptive = wr.params(n_times=512)
    # half_nbw = 4.0 * 512 / 1000 ≈ 2.048; n_tapers_max = floor(2*2.048 - 1) = 3
    assert window_fun.shape[1] == 512
    assert len(eig) >= 1


def test_params_none_bandwidth_uses_default():
    """``self.bandwidth is None`` defaults half_nbw to 4 (line 153-154)."""
    wr = _wr(bandwidth=None)
    window_fun, eig, adaptive = wr.params(n_times=512)
    assert window_fun.shape[1] == 512


def test_params_raises_on_bandwidth_too_low():
    """half_nbw < 0.5 → ValueError (line 155-159)."""
    wr = _wr(bandwidth=0.1)  # 0.1 * 512 / 1000 = 0.0512 < 0.5
    with pytest.raises(ValueError, match="yields a normalized bandwidth"):
        wr.params(n_times=512)


def test_params_adaptive_disabled_when_few_tapers():
    """``adaptive=True`` but fewer than 3 tapers → warn and disable
    (line 168-172).

    Use a regime where half_nbw lands in (0.5, 1.5) → n_tapers_max ∈
    {0, 1, 2}, ensuring < 3 tapers are produced.
    """
    # bandwidth=2.0 * 512 / 1000 = 1.024 → n_tapers_max = floor(2*1.024 -1) = 1
    wr = _wr(bandwidth=2.0, adaptive=True, low_bias=True)
    _, eig, adaptive = wr.params(n_times=512)
    assert len(eig) < 3
    assert adaptive is False
    assert wr.adaptive is False


# -------------------------------------------------------------------- #
# get_thresh — caches and returns (window_fun, threshold)
# -------------------------------------------------------------------- #

def test_get_thresh_returns_threshold_for_explicit_n_times():
    # Need enough samples that bandwidth*n_times/sfreq yields a half_nbw
    # large enough for at least 1 taper.
    wr = _wr(bandwidth=5.0, sfreq=500.0)
    # 5 * 256 / 500 = 2.56 → n_tapers_max = 4
    window_fun, thresh = wr.get_thresh(n_times=256)
    assert window_fun.shape[1] == 256
    assert thresh > 0


def test_get_thresh_uses_filter_length_when_n_times_none():
    # 5 * 1024 / 1000 = 5.12 → n_tapers_max = 9.
    wr = _wr(bandwidth=5.0, filter_length=1024)
    window_fun, thresh = wr.get_thresh(None)
    assert window_fun.shape[1] == 1024


# -------------------------------------------------------------------- #
# spectra
# -------------------------------------------------------------------- #

def test_spectra_shapes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 256))
    dpss = np.ones((4, 256))  # placeholder tapers
    # Normalise tapers so spectra is meaningful.
    dpss /= np.sqrt(np.sum(dpss ** 2, axis=1, keepdims=True))
    x_mt, freqs = spectra(x, dpss, sfreq=500.0)
    # Output shape = (..., n_tapers, n_freqs)
    assert x_mt.shape == (3, 4, 256 // 2 + 1)
    assert freqs.shape == (256 // 2 + 1,)
    # Frequencies span DC to Nyquist.
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(250.0)


def test_spectra_does_not_mutate_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 128))
    original = x.copy()
    dpss = np.ones((1, 128))
    dpss /= np.sqrt(np.sum(dpss ** 2, axis=1, keepdims=True))
    _ = spectra(x, dpss, sfreq=500.0)
    np.testing.assert_array_equal(x, original)


def test_spectra_n_fft_default_uses_input_length():
    x = np.zeros((1, 100))
    dpss = np.ones((1, 100))
    dpss /= np.sqrt(np.sum(dpss ** 2, axis=1, keepdims=True))
    x_mt, freqs = spectra(x, dpss, sfreq=200.0)
    # n_fft defaulted to 100 → rfft length = 100//2 + 1.
    assert freqs.shape == (51,)


def test_spectra_explicit_n_fft():
    x = np.zeros((1, 100))
    dpss = np.ones((1, 100))
    dpss /= np.sqrt(np.sum(dpss ** 2, axis=1, keepdims=True))
    x_mt, freqs = spectra(x, dpss, sfreq=200.0, n_fft=128)
    # Note: rfft output length is n_fft//2 + 1 (= 65), but freqs is
    # computed from the explicit n_fft.
    assert freqs.shape == (65,)


# -------------------------------------------------------------------- #
# _mt_remove — scalar notch_widths conversion (line 268-270)
# -------------------------------------------------------------------- #

def test_mt_remove_promotes_scalar_notch_width():
    """When ``notch_widths`` is a scalar number, the function broadcasts
    it to one width per line frequency (line 268-270)."""
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    n = 1024
    t = np.arange(n) / sfreq
    # Pure 60 Hz tone (plus tiny noise so the F-stat is finite).
    x = np.cos(2 * np.pi * 60 * t) + 0.01 * rng.standard_normal(n)
    wr = _wr(sfreq=sfreq, line_freqs=[60.0], filter_length=n,
             bandwidth=4.0, low_bias=True, adaptive=False)
    window_fun, thresh = wr.get_thresh(n_times=n)
    # Pass a SCALAR notch_width — must be broadcast to match line_freqs.
    x_clean, found = _mt_remove(x, sfreq, [60.0], notch_widths=2.0,
                                window_fun=window_fun, threshold=thresh,
                                get_thresh=wr.get_thresh)
    assert x_clean.shape == x.shape


def test_mt_remove_no_line_freqs_returns_all_found():
    """When ``line_freqs`` is None, ``notch_widths`` filtering is
    skipped entirely."""
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    n = 512
    x = rng.standard_normal(n)
    wr = _wr(sfreq=sfreq, line_freqs=None, filter_length=n,
             bandwidth=4.0)
    window_fun, thresh = wr.get_thresh(n_times=n)
    x_clean, found = _mt_remove(x, sfreq, None, notch_widths=None,
                                window_fun=window_fun, threshold=thresh,
                                get_thresh=wr.get_thresh)
    assert x_clean.shape == x.shape


def test_mt_remove_window_size_mismatch_refits():
    """When ``x.shape[-1] != window_fun.shape[-1]``, ``get_thresh`` is
    called to refit (line ~256-257)."""
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    wr = _wr(sfreq=sfreq, line_freqs=[60.0], filter_length=1024,
             bandwidth=4.0, low_bias=True, adaptive=False)
    # Build with default n=1024, then call with a different size.
    window_fun, thresh = wr.get_thresh(n_times=1024)
    x = rng.standard_normal(512)  # half the size
    x_clean, _ = _mt_remove(x, sfreq, None, None,
                            window_fun=window_fun, threshold=thresh,
                            get_thresh=wr.get_thresh)
    assert x_clean.shape == x.shape
