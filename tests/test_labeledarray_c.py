"""Pytest coverage for the C-extension ``LabeledArray`` class.

Originally a print-based smoke script; now expressed as ``test_*``
functions so it runs as part of the battery. These tests target the
hand-C class directly (``ieeg.arrays.labeledarray.LabeledArray``)
without the Python ``label.py`` subclass on top.
"""
from __future__ import annotations

import numpy as np
import pytest

from ieeg.arrays.labeledarray import LabeledArray


LABS_2D = (("r0", "r1"), ("c0", "c1", "c2"))


def _arr2d():
    return LabeledArray(np.arange(6, dtype=np.float32).reshape(2, 3),
                        labels=LABS_2D)


# -------------------------------------------------------------------- #
# Indexing
# -------------------------------------------------------------------- #

def test_row_label_slice():
    a = _arr2d()
    s = a["r0"]
    assert s.shape == (3,)
    assert tuple(s.labels[0]) == ("c0", "c1", "c2")
    np.testing.assert_array_equal(np.asarray(s),
                                  np.array([0, 1, 2], dtype=np.float32))


def test_column_tuple_label_slice():
    a = _arr2d()
    s2 = a[:, ("c1", "c2")]
    assert s2.shape == (2, 2)
    assert tuple(s2.labels[1]) == ("c1", "c2")
    np.testing.assert_array_equal(
        np.asarray(s2),
        np.array([[1, 2], [4, 5]], dtype=np.float32))


def test_ellipsis_after_string_key():
    a = _arr2d()
    s3 = a["r1", ...]
    assert s3.shape == (3,)
    assert tuple(s3.labels[0]) == ("c0", "c1", "c2")


def test_newaxis_with_string_key():
    a = _arr2d()
    s4 = a[np.newaxis, "r0"]
    assert s4.shape == (1, 3)
    # The new axis gets a default singleton label; the surviving
    # column axis keeps its labels.
    assert tuple(s4.labels[-1]) == ("c0", "c1", "c2")


# -------------------------------------------------------------------- #
# Views: transpose / iter
# -------------------------------------------------------------------- #

def test_transpose_preserves_label_block():
    # The C class's ``.T`` returns a transposed numpy view but does NOT
    # rotate the label block (legacy behaviour; the Python subclass in
    # ``label.py`` does rotate via ``__array_finalize__``). The
    # contract here is just that the labels remain accessible and the
    # transposed shape is correct.
    a = _arr2d()
    t = a.T
    assert t.shape == (3, 2)
    assert hasattr(t, "labels")
    # All original label tokens still reachable.
    flat = tuple(x for axis in t.labels for x in axis)
    for tok in ("r0", "r1", "c0", "c1", "c2"):
        assert tok in flat


def test_iter_yields_per_row():
    a = _arr2d()
    rows = list(a)
    assert len(rows) == 2
    for row in rows:
        assert row.shape == (3,)


# -------------------------------------------------------------------- #
# Item assignment
# -------------------------------------------------------------------- #

def test_assign_row_with_list():
    a = _arr2d()
    a[0] = [2, 3, 4]
    np.testing.assert_array_equal(np.asarray(a[0]),
                                  np.array([2, 3, 4], dtype=np.float32))


def test_assign_row_with_ndarray_then_iadd():
    a = _arr2d()
    a[1] = np.array([5, 6, 7])
    a[1] += 1
    np.testing.assert_array_equal(np.asarray(a[1]),
                                  np.array([6, 7, 8], dtype=np.float32))


# -------------------------------------------------------------------- #
# Reshape / combine
# -------------------------------------------------------------------- #

def test_combine_concatenates_labels_along_axis():
    a = _arr2d()
    out = a.combine((0, 1))
    # Combine ``(0, 1)`` collapses the first two axes; the resulting
    # labels are the cross-product of (r0,r1) × (c0,c1,c2) joined by
    # the delimiter ('-').
    assert out.shape == (6,)
    joined = tuple(out.labels[0])
    expected = (
        "r0-c0", "r0-c1", "r0-c2",
        "r1-c0", "r1-c1", "r1-c2",
    )
    assert joined == expected


# -------------------------------------------------------------------- #
# Weighted helper (regression: in-place ufuncs + masks + dtype kwarg)
# -------------------------------------------------------------------- #

def _weighted_preserve_stats(data, weights, axis=None):
    where = ~np.isnan(data)
    kwargs = {"where": where, "dtype": "f4"}
    orig_mean = np.mean(data, **kwargs)
    orig_std = np.std(data, **kwargs)
    if axis is None:
        data *= weights
    else:
        data *= weights.reshape(
            [1 if i != axis else -1 for i in range(data.ndim)])
    weighted_mean = np.mean(data, **kwargs)
    weighted_std = np.std(data, **kwargs)
    data -= weighted_mean
    data *= orig_std / weighted_std
    data += orig_mean
    return data


def test_weighted_preserve_stats_keeps_mean_and_std():
    a = _arr2d()
    weights = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    orig = np.asarray(a).copy()
    out = _weighted_preserve_stats(a, weights)
    # In-place mutation; the LabeledArray storage is now scaled. The
    # contract is "preserve mean and std under the where=~isnan mask".
    np.testing.assert_allclose(
        float(np.mean(np.asarray(out))),
        float(np.mean(orig)),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(np.std(np.asarray(out))),
        float(np.std(orig)),
        atol=1e-5,
    )


# -------------------------------------------------------------------- #
# C-class survives a faulthandler-tracked stress test (no crash)
# -------------------------------------------------------------------- #

def test_c_class_stress_no_crash():
    # Many small ops in a row — the original print-script exposed
    # crashes under the C extension with a 10-second faulthandler.
    a = _arr2d()
    for _ in range(50):
        _ = a.T
        _ = a["r0"]
        _ = a[:, "c0"]
        _ = a[np.newaxis, "r0"]
        _ = a.combine((0, 1))
