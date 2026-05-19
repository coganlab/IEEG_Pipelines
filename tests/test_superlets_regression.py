"""Byte-equivalence regression tests for the superlets CPU kernel migration.

Reference fixtures live in ``tests/fixtures/superlets_*.npz`` (produced
by ``tests/fixtures/generate_superlets_fixtures.py`` from the previous
Numba implementation). After the pure-C OpenMP rewrite, these tests
assert that outputs match the reference within float64 ULP tolerances.
"""
from pathlib import Path

import numpy as np
import pytest

from ieeg.timefreq import _superlets_kernels


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cx_fixtures():
    return dict(np.load(FIXTURE_DIR / "superlets_cxmorelet_batch.npz"))


@pytest.fixture(scope="module")
def mg_fixtures():
    return dict(np.load(FIXTURE_DIR / "superlets_mask_geomean.npz"))


@pytest.mark.parametrize("case", [0, 1])
def test_cxmorelet_batch(cx_fixtures, case):
    freqs = cx_fixtures[f"case_{case}_freqs"]
    cycles = cx_fixtures[f"case_{case}_cycles"]
    sampling_freq = float(cx_fixtures[f"case_{case}_sampling_freq"])
    expected = cx_fixtures[f"case_{case}_wavelets"]

    out = np.zeros_like(expected)
    _superlets_kernels.cxmorelet_batch(
        np.ascontiguousarray(freqs, dtype=np.float64),
        np.ascontiguousarray(cycles, dtype=np.float64),
        sampling_freq,
        out,
    )

    # The kernel performs the same math as the Numba version. Both use
    # libm exp/cos/sin/sqrt with float64 accumulators. Differences would
    # arise only from rounding in the normalization step (division by
    # the sum of absolute values); these are sub-ULP for typical inputs.
    np.testing.assert_allclose(
        out, expected, rtol=1e-12, atol=1e-15,
        err_msg=f"cxmorelet_batch case {case}",
    )


@pytest.mark.parametrize("case", [0, 1, 2])
def test_apply_mask_and_geomean(mg_fixtures, case):
    arr = mg_fixtures[f"case_{case}_input"].copy()
    orders = mg_fixtures[f"case_{case}_orders"].astype(np.int64)
    eps = float(mg_fixtures[f"case_{case}_eps"])
    expected_result = mg_fixtures[f"case_{case}_result"]
    expected_after = mg_fixtures[f"case_{case}_input_after"]

    result = np.zeros_like(expected_result)
    _superlets_kernels.apply_mask_and_geomean(arr, orders, eps, result)

    # Result: geometric mean derived from log/exp. Sub-ULP rounding.
    np.testing.assert_allclose(
        result, expected_result, rtol=1e-12, atol=1e-15,
        err_msg=f"apply_mask_and_geomean result case {case}",
    )
    # In-place mutation of `out` should be bit-identical (just sets to 1.0).
    np.testing.assert_array_equal(
        arr, expected_after,
        err_msg=f"apply_mask_and_geomean mutated input case {case}",
    )
