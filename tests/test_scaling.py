"""Coverage for ``ieeg.calc.scaling.rescale``.

The base ``np.ndarray`` dispatch supports 6 baseline-correction modes
('mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio') plus
an error case for unknown modes. The MNE-typed dispatches (BaseEpochs,
EpochsTFR, AverageTFR) are integration paths that require building MNE
objects from scratch.
"""
from __future__ import annotations

import numpy as np
import pytest

from ieeg.calc.scaling import rescale, _log_rescale


# -------------------------------------------------------------------- #
# _log_rescale (tiny helper, both branches)
# -------------------------------------------------------------------- #

def test_log_rescale_with_baseline():
    msg = _log_rescale(object(), mode="mean")
    assert "baseline correction" in msg
    assert "mean" in msg


def test_log_rescale_without_baseline():
    msg = _log_rescale(None)
    assert "No baseline correction" in msg


# -------------------------------------------------------------------- #
# rescale: every mode, deterministic baseline
# -------------------------------------------------------------------- #

@pytest.fixture
def two_axis_data():
    rng = np.random.default_rng(0)
    # Shape (channels, time). Baseline is the same noise but offset by
    # +5 so the modes have well-defined effects.
    base = rng.standard_normal((4, 32)).astype(np.float64) + 5.0
    data = rng.standard_normal((4, 32)).astype(np.float64) + 10.0
    return data, base


def test_rescale_mean_subtracts_baseline_mean(two_axis_data):
    data, base = two_axis_data
    out = rescale(data.copy(), base, mode="mean", copy=False, axis=-1)
    # mean of base along axis=-1.
    expected_mean = base.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, data - expected_mean)


def test_rescale_ratio_divides_by_baseline_mean(two_axis_data):
    data, base = two_axis_data
    out = rescale(data.copy(), base, mode="ratio", copy=True, axis=-1)
    expected = data / base.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, expected)


def test_rescale_copy_does_not_mutate_input(two_axis_data):
    data, base = two_axis_data
    original = data.copy()
    out = rescale(data, base, mode="mean", copy=True, axis=-1)
    np.testing.assert_array_equal(data, original)
    assert out is not data


def test_rescale_in_place_mutates_input(two_axis_data):
    data, base = two_axis_data
    original = data.copy()
    out = rescale(data, base, mode="mean", copy=False, axis=-1)
    # Same object back.
    assert out is data
    # Mutation visible on original buffer.
    assert not np.array_equal(data, original)


def test_rescale_logratio_then_log10(two_axis_data):
    data, base = two_axis_data
    out = rescale(data.copy(), base, mode="logratio", copy=False, axis=-1)
    expected = np.log10(data / base.mean(axis=-1, keepdims=True))
    np.testing.assert_allclose(out, expected)


def test_rescale_percent(two_axis_data):
    data, base = two_axis_data
    out = rescale(data.copy(), base, mode="percent", copy=False, axis=-1)
    m = base.mean(axis=-1, keepdims=True)
    expected = (data - m) / m
    np.testing.assert_allclose(out, expected)


def test_rescale_zscore(two_axis_data):
    data, base = two_axis_data
    out = rescale(data.copy(), base, mode="zscore", copy=False, axis=-1)
    m = base.mean(axis=-1, keepdims=True)
    # dist uses ddof=1 in scaling.rescale; std with ddof=1 matches.
    s = base.std(axis=-1, keepdims=True, ddof=1)
    expected = (data - m) / s
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_rescale_zlogratio(two_axis_data):
    data, base = two_axis_data
    out = rescale(data.copy(), base, mode="zlogratio", copy=False, axis=-1)
    m = base.mean(axis=-1, keepdims=True)
    # std of the BASELINE under log? scaling.rescale uses the std
    # computed on the unlogged baseline (see fun definition for
    # zlogratio: divide by m, log10, then divide by s).
    s = base.std(axis=-1, keepdims=True, ddof=1)
    expected = np.log10(data / m) / s
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_rescale_unknown_mode_raises():
    data = np.ones((3, 10))
    base = np.ones((3, 10)) * 2.0
    with pytest.raises(NotImplementedError):
        rescale(data, base, mode="not-a-mode")


# -------------------------------------------------------------------- #
# rescale axis variations
# -------------------------------------------------------------------- #

def test_rescale_axis_tuple(two_axis_data):
    data, base = two_axis_data
    # When axis is a tuple, dist reduces over those axes simultaneously.
    out = rescale(data.copy(), base, mode="mean", copy=False, axis=(0, -1))
    expected_mean = base.mean(axis=(0, -1), keepdims=True)
    np.testing.assert_allclose(out, data - expected_mean)


def test_rescale_3d_data():
    rng = np.random.default_rng(42)
    base = rng.standard_normal((2, 3, 50)) + 5.0
    data = rng.standard_normal((2, 3, 50))
    out = rescale(data.copy(), base, mode="mean", axis=-1)
    assert out.shape == data.shape
