"""Byte-equivalence regression tests for the hilbert migration.

Generated reference outputs live in ``tests/fixtures/hilbert_*.npz``
(produced by ``tests/fixtures/generate_hilbert_fixtures.py`` from the
previous Cython implementation). After the pure-Python + OpenMP-kernel
rewrite, these tests assert that outputs match the reference within
float32 ULP tolerances.

If these fail after an intentional algorithm change, regenerate the
fixtures by re-running the generator script, bump the version constant,
and review the diff carefully.
"""
from pathlib import Path

import numpy as np
import pytest

from ieeg.timefreq.hilbert import (
    get_centers,
    filterbank_hilbert_first_half_wrapper,
    extract_channel_wrapper,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def gc_fixtures():
    return dict(np.load(FIXTURE_DIR / "hilbert_get_centers.npz"))


@pytest.fixture(scope="module")
def fh_fixtures():
    return dict(np.load(FIXTURE_DIR / "hilbert_first_half.npz"))


@pytest.fixture(scope="module")
def ec_fixtures():
    return dict(np.load(FIXTURE_DIR / "hilbert_extract_channel.npz"))


# Tolerance philosophy
# --------------------
# The previous Cython sometimes computed intermediate steps in float32
# where pure Python (CPython) keeps float64 until an explicit cast,
# producing ~1-ULP drift in float32 outputs. We allow tolerances
# corresponding to float32 precision (~7 decimal digits):
# - Tight: outputs derived from a single arithmetic chain (rtol 1e-6).
# - Loose: outputs that accumulate through FFT + IFFT + complex
#   multiplies (rtol 1e-4).


@pytest.mark.parametrize("case", [0, 1, 2, 3])
def test_get_centers(gc_fixtures, case):
    args = gc_fixtures[f"case_{case}_args"]
    expected = gc_fixtures[f"case_{case}_cfs"]
    actual = get_centers((float(args[0]), float(args[1])),
                         octSpace=float(args[2]))
    # Pure-Python float64 intermediates land at the same final value
    # within float32 ULP precision after the trailing .astype('float32').
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=1e-5,
        err_msg=f"get_centers case {case}",
    )


@pytest.mark.parametrize("case", [0, 1])
def test_first_half_wrapper(fh_fixtures, case):
    x = fh_fixtures[f"case_{case}_x"]
    fs = int(fh_fixtures[f"case_{case}_fs"])
    minf = float(fh_fixtures[f"case_{case}_minf"])
    maxf = float(fh_fixtures[f"case_{case}_maxf"])
    oct_space = float(fh_fixtures[f"case_{case}_oct_space"])

    Xf, freqs, cfs, N, sds, h = filterbank_hilbert_first_half_wrapper(
        x, fs, minf, maxf, oct_space
    )

    # N is an integer scalar.
    assert int(N) == int(fh_fixtures[f"case_{case}_N"])

    # freqs is a deterministic arange — bit-exact.
    np.testing.assert_array_equal(np.asarray(freqs),
                                  fh_fixtures[f"case_{case}_freqs"])

    # h is built from integer constants cast to complex64 — bit-exact.
    np.testing.assert_array_equal(np.asarray(h),
                                  fh_fixtures[f"case_{case}_h"])

    # cfs / sds carry the get_centers precision drift.
    np.testing.assert_allclose(np.asarray(cfs),
                               fh_fixtures[f"case_{case}_cfs"],
                               rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(np.asarray(sds),
                               fh_fixtures[f"case_{case}_sds"],
                               rtol=1e-6, atol=1e-5)

    # Xf comes from scipy.fft + .astype('complex64'). FFT routines may
    # have minor reordering vs the previous implementation; allow ULP
    # tolerance.
    np.testing.assert_allclose(np.asarray(Xf),
                               fh_fixtures[f"case_{case}_Xf"],
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", [0, 1])
def test_extract_channel_wrapper(ec_fixtures, case):
    x = ec_fixtures[f"case_{case}_x"]
    fs = int(ec_fixtures[f"case_{case}_fs"])
    minf = float(ec_fixtures[f"case_{case}_minf"])
    maxf = float(ec_fixtures[f"case_{case}_maxf"])
    oct_space = float(ec_fixtures[f"case_{case}_oct_space"])
    expected = ec_fixtures[f"case_{case}_hilb_amp"]  # (n_ch, N, n_filt_in_band)

    Xf, freqs, cfs, N, sds, h = filterbank_hilbert_first_half_wrapper(
        x, fs, minf, maxf, oct_space
    )
    per_chan = []
    for ch in range(Xf.shape[1]):
        hilb_amp = extract_channel_wrapper(
            Xf[:, ch], freqs, cfs, int(N), sds, h,
            np.float32(minf), np.float32(maxf),
        )
        per_chan.append(np.asarray(hilb_amp))
    actual = np.stack(per_chan, axis=0)

    # Full pipeline: FFT → gaussian filter bank → complex multiply →
    # inverse FFT → |.|. Accumulated rounding through these steps
    # warrants the looser tolerance.
    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=1e-6,
        err_msg=f"extract_channel_wrapper case {case}",
    )
