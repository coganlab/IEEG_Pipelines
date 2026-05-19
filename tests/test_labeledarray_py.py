"""Pytest coverage for the Python ``LabeledArray`` (numpy-backed) class.

Originally a print-based smoke script; this file now asserts the same
behaviours as proper ``test_*`` functions so they're collected and run.
The Python class (``ieeg.arrays.label.LabeledArray``) is a thin subclass
of the C-extension class (``ieeg.arrays.labeledarray.LabeledArray``)
augmented with the shared ``_LabeledMixin`` for label semantics.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from ieeg.arrays.labeledarray import LabeledArray as cla_c  # C impl
from ieeg.arrays.label import LabeledArray as cla  # Python subclass


# Fixture shape used across the original print-script.
LABELS_3D = (("a", "b"), ("c", "d", "e"), ("f", "g", "h", "i"))


# Common helper: build the canonical 2x3x4 array
def _ones():
    return cla(np.ones((2, 3, 4), dtype=float), LABELS_3D)


# -------------------------------------------------------------------- #
# Construction & basic properties
# -------------------------------------------------------------------- #

def test_construct_shape_and_labels():
    la = _ones()
    assert la.shape == (2, 3, 4)
    # Mixin presents labels as a list-of-lists for the repr path; the
    # important contract is positional equality, axis by axis.
    for axis, expected in enumerate(LABELS_3D):
        assert tuple(la.labels[axis]) == expected


def test_iter_yields_per_row():
    la = _ones()
    rows = [a for a in la]
    assert len(rows) == 2
    for row in rows:
        assert row.shape == (3, 4)


def test_repr_contains_label_block():
    la = _ones()
    # The repr embeds the labels via the mixin's _label_formatter.
    s = repr(la)
    for lab in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        assert lab in s


# -------------------------------------------------------------------- #
# Item assignment & retrieval
# -------------------------------------------------------------------- #

def test_setitem_scalar_path():
    la = _ones()
    la["a", "c", "f"] = 2.0
    assert la["a", "c", "f"] == 2.0


def test_setitem_vector_path():
    la = _ones()
    la["a", "d"] = np.array([3, 3, 3, 3])
    np.testing.assert_array_equal(np.asarray(la["a", "d"]),
                                  np.array([3.0, 3.0, 3.0, 3.0]))


def test_get_partial_path_returns_subarray():
    la = _ones()
    la["a", "c", "f"] = 2.0
    out = la["a", "c"]
    assert out.shape == (4,)
    np.testing.assert_array_equal(np.asarray(out),
                                  np.array([2.0, 1.0, 1.0, 1.0]))


def test_tuple_in_tuple_indexing():
    la = _ones()
    la["a", "c", "f"] = 2.0
    la["a", "d"] = np.array([3.0, 3.0, 3.0, 3.0])
    out = la[("a", "b"), :]
    assert out.shape == (2, 3, 4)
    # First row remembers the writes; second row is unchanged.
    assert out[0, 0, 0] == 2.0
    assert out[0, 1, 0] == 3.0
    assert out[1, 0, 0] == 1.0


# -------------------------------------------------------------------- #
# find / take / take_along_axis
# -------------------------------------------------------------------- #

def test_find_axis0():
    la = _ones()
    idx = la.find("a", 0)
    assert idx == 0


def test_np_take_by_label():
    la = _ones()
    out = np.take(la, np.array(["f", "g"]), axis=2)
    assert out.shape == (2, 3, 2)
    # Take with labels must still produce a LabeledArray and the
    # selected labels match what was asked for.
    assert tuple(out.labels[2]) == ("f", "g")


def test_np_take_along_axis():
    la = _ones()
    out = np.take_along_axis(la[:, :, 0],
                             np.array([[0, 0, 0], [1, 1, 1]]),
                             axis=0)
    assert out.shape == (2, 3)


# -------------------------------------------------------------------- #
# np.* shape ops
# -------------------------------------------------------------------- #

def test_concatenate_two_slices():
    la = _ones()
    out = np.concatenate((la["a"], la["b"]), axis=0)
    assert out.shape == (6, 4)


def test_swapaxes_and_transpose():
    la = _ones()
    out1 = np.swapaxes(la, 0, 1)
    out2 = np.transpose(la, (1, 0, 2))
    assert out1.shape == out2.shape == (3, 2, 4)


# -------------------------------------------------------------------- #
# NaN handling: dropna / nanmean
# -------------------------------------------------------------------- #

def test_dropna_label_subset():
    la = _ones()
    la[:, :, 0] = float("nan")
    out = la.dropna()
    # The all-NaN slice along axis 2 (index 0) should be dropped, so
    # the surviving labels are ('g','h','i').
    assert tuple(out.labels[2]) == ("g", "h", "i")


def test_dropna_after_nanmean():
    la = _ones()
    la[:, :, 0] = float("nan")
    out = np.nanmean(la, axis=(0, 1)).dropna()
    # Reducing over axes 0+1 leaves axis 2; the all-NaN entry drops.
    assert tuple(out.labels[0]) == ("g", "h", "i")


def test_mean_with_where_mask():
    la = _ones()
    la[:, :, 0] = float("nan")
    # Mean over (0,1) with where=isnan keeps the all-NaN column.
    out = np.mean(la, axis=(0, 1), where=np.isnan(la))
    # axis 2 survives with 4 entries.
    assert out.shape == (4,)


# -------------------------------------------------------------------- #
# fromfile/tofile round-trip
# -------------------------------------------------------------------- #

def test_fromfile_round_trip(tmp_path):
    # Use tmp_path so each test run is isolated.
    target = str(tmp_path / "tmp_la")
    la = _ones()
    la.tofile(target)
    la2 = cla.fromfile(target)
    assert la2.shape == la.shape
    # Persisting on disk does NOT preserve the original dtype here
    # (legacy behaviour: stored as str inside .npz labels). Just
    # check value equality.
    np.testing.assert_array_equal(np.asarray(la2),
                                  np.asarray(la))


# -------------------------------------------------------------------- #
# In-place ops & meshgrid
# -------------------------------------------------------------------- #

def test_inplace_add_preserves_labels():
    la = _ones()
    la += 1.0
    assert tuple(la.labels[0]) == ("a", "b")
    np.testing.assert_array_equal(np.asarray(la),
                                  np.full((2, 3, 4), 2.0))


def test_ix_indexing_matches_simple_slice():
    la = _ones()
    out_ix = la[np.ix_(*(np.arange(2) for _ in la.labels))]
    out_ctrl = la[:2, :2, :2]
    np.testing.assert_array_equal(np.asarray(out_ix),
                                  np.asarray(out_ctrl))


# -------------------------------------------------------------------- #
# from_dict / to_dict
# -------------------------------------------------------------------- #

def test_from_dict_round_trip_with_nans():
    data = {"a": {"b": {"c": 1.0, "d": np.nan}}}
    ad = cla.from_dict(data)
    out = ad.dropna()
    assert out.shape == (1, 1, 1)


# -------------------------------------------------------------------- #
# Indexing matrix — every key form from the original script
# -------------------------------------------------------------------- #

@pytest.mark.parametrize("idx,expected_shape", [
    ((0,), (3, 4)),
    (("a",), (3, 4)),
    ((("a", "b"),), (2, 3, 4)),
    ((slice(None),), (2, 3, 4)),
    ((slice(None), slice(None), slice(None)), (2, 3, 4)),
    ((slice(None), slice(None), 0), (2, 3)),
    ((slice(None), -1, slice(None)), (2, 4)),
    (("a", slice(None), slice(None)), (3, 4)),
    (("a", slice(None), 0), (3,)),
    # 'a' + ('f','g') are both treated as fancy indices that broadcast
    # together (1 ⊗ 2 → 2), and a basic slice on axis 1 stays separate.
    # Two non-adjacent fancy indices push the advanced result to the
    # front → (2, 3) rather than the "naive" (3, 2).
    (("a", slice(None), ("f", "g")), (2, 3)),
])
def test_indexing_keys(idx, expected_shape):
    la = _ones()
    out = la[idx]
    assert out.shape == expected_shape


# -------------------------------------------------------------------- #
# Both class types expose ``.labels`` consistently
# -------------------------------------------------------------------- #

def test_c_class_labels_attr():
    la3 = cla_c(np.random.rand(2, 3),
                labels=[("a", "b"), ("c", "d", "e")])
    # C class stores labels on the object too.
    assert la3.shape == (2, 3)
    # Walks through __dict__ — should not crash.
    _ = la3.__dict__


def test_py_class_labels_attr():
    la4 = cla(np.random.rand(2, 3),
              labels=[("a", "b"), ("c", "d", "e")])
    assert la4.shape == (2, 3)
    assert tuple(la4.labels[0]) == ("a", "b")
    _ = la4.__dict__
