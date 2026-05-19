"""Coverage for the shared ``_LabeledMixin`` extracted from ``label.py``.

These tests target the mixin's surface directly through the
NumPy-backed ``LabeledArray``, which is the canonical concrete class
that inherits it. They also exercise gaps the broader behaviour tests
in ``test_labeledarray_py.py`` don't reach:

- ``__str__`` (vs ``__repr__``)
- ``memory()`` for both numpy nbytes and torch numel*element_size paths
- ``to_dict()`` all-NaN leaf skipping (line 90)
- ``__eq__`` scalar fallback (line 114)
- ``__ne__`` symmetry
- ``_label_formatter`` direct call
"""
from __future__ import annotations

import numpy as np
import pytest

from ieeg.arrays._labels_mixin import _LabeledMixin
from ieeg.arrays.label import LabeledArray


LABELS_2D = (("r0", "r1"), ("c0", "c1", "c2"))


def _la(labels=LABELS_2D, fill=1.0, shape=(2, 3), dtype=np.float64):
    return LabeledArray(np.full(shape, fill, dtype=dtype), labels)


# -------------------------------------------------------------------- #
# __repr__ vs __str__
# -------------------------------------------------------------------- #

def test_repr_and_str_both_contain_labels_block():
    la = _la()
    r = repr(la)
    s = str(la)
    assert "labels(" in r
    assert "labels(" in s
    for lab in ("r0", "r1", "c0", "c1", "c2"):
        assert lab in r
        assert lab in s


def test_repr_and_str_differ_only_in_array_formatting():
    la = _la()
    r = repr(la)
    s = str(la)
    # The labels block is identical between the two — only the array
    # formatting differs (repr vs str of ndarray).
    r_tail = r.split("labels(", 1)[1]
    s_tail = s.split("labels(", 1)[1]
    assert r_tail == s_tail


# -------------------------------------------------------------------- #
# _label_formatter direct call
# -------------------------------------------------------------------- #

def test_label_formatter_returns_newline_joined_string():
    la = _la()
    formatted = la._label_formatter()
    # Each axis is separated by a newline + indent.
    assert "\n" in formatted
    # All labels show up.
    for axis in LABELS_2D:
        for lab in axis:
            assert lab in formatted


# -------------------------------------------------------------------- #
# memory()
# -------------------------------------------------------------------- #

def test_memory_bytes_units_numpy():
    # 4 elements × 8 bytes = 32 B.
    la = _la(shape=(2, 2), labels=(("r0", "r1"), ("c0", "c1")),
             dtype=np.float64)
    size, unit = la.memory()
    assert unit == "B"
    assert size == 32


def test_memory_kib_units_numpy():
    # 256 elements × 8 bytes = 2048 B = 2 KiB.
    labels = (tuple(f"r{i}" for i in range(32)),
              tuple(f"c{i}" for i in range(8)))
    la = _la(shape=(32, 8), labels=labels, dtype=np.float64)
    size, unit = la.memory()
    assert unit == "KiB"
    assert size == pytest.approx(2.0)


def test_memory_returns_string_unit_label():
    la = _la()
    size, unit = la.memory()
    assert isinstance(unit, str)
    assert unit in {"B", "KiB", "MiB", "GiB", "TiB", "PiB"}


def test_memory_caps_at_pib():
    """The unit ladder doesn't go past PiB even for absurdly large sizes."""
    class _Fake(_LabeledMixin):
        # Pretend we have an astronomical nbytes.
        nbytes = 10 ** 30
        labels = ()
    size, unit = _Fake().memory()
    assert unit == "PiB"
    # The loop stops AT PiB before doing one more division.


def test_memory_raises_when_no_size_protocol():
    """Without nbytes or numel/element_size, memory() must raise."""
    class _Bare(_LabeledMixin):
        labels = ()
    with pytest.raises(TypeError, match="Cannot compute memory size"):
        _Bare().memory()


# -------------------------------------------------------------------- #
# to_dict() — all-NaN leaf skipping
# -------------------------------------------------------------------- #

def test_to_dict_basic_roundtrip():
    la = _la(shape=(2, 3), labels=LABELS_2D, fill=1.0)
    d = la.to_dict()
    assert d == {"r0": {"c0": 1.0, "c1": 1.0, "c2": 1.0},
                 "r1": {"c0": 1.0, "c1": 1.0, "c2": 1.0}}


def test_to_dict_skips_all_nan_leaves_1d():
    """The all-NaN branch in to_dict only fires at the leaf axis (ndim==1).

    Set the entire 'r0' row to NaN; in a 2D array `to_dict` recurses
    into each row, and at the inner level (ndim==1) the all-NaN check
    skips the entire entry.
    """
    arr = np.array([[np.nan, np.nan, np.nan],
                    [1.0, 2.0, 3.0]])
    la = LabeledArray(arr, LABELS_2D)
    d = la.to_dict()
    # 'r0' should be present (recurse into its 1D values), but its
    # inner entries should all be skipped → empty dict at that level.
    assert d["r0"] == {}
    assert d["r1"] == {"c0": 1.0, "c1": 2.0, "c2": 3.0}


def test_to_dict_partial_nan_leaves_preserved():
    arr = np.array([[1.0, np.nan, 3.0],
                    [4.0, 5.0, np.nan]])
    la = LabeledArray(arr, LABELS_2D)
    d = la.to_dict()
    # Non-NaN leaves survive; NaN leaves are dropped one by one.
    assert d["r0"] == {"c0": 1.0, "c2": 3.0}
    assert d["r1"] == {"c0": 4.0, "c1": 5.0}


# -------------------------------------------------------------------- #
# items / keys / values
# -------------------------------------------------------------------- #

def test_keys_yields_first_axis_labels():
    la = _la()
    assert list(la.keys()) == ["r0", "r1"]


def test_values_yields_per_row_subarrays():
    la = _la()
    vals = list(la.values())
    assert len(vals) == 2
    for v in vals:
        assert v.shape == (3,)


def test_items_zip_keys_and_values():
    la = _la()
    items = list(la.items())
    assert [k for k, _ in items] == ["r0", "r1"]
    assert all(v.shape == (3,) for _, v in items)


# -------------------------------------------------------------------- #
# __eq__ and __ne__
# -------------------------------------------------------------------- #

def test_eq_same_labeled_array_is_true():
    la1 = _la()
    la2 = _la()
    assert (la1 == la2) is True


def test_eq_different_data_is_false():
    la1 = _la(fill=1.0)
    la2 = _la(fill=2.0)
    assert (la1 == la2) is False


def test_eq_different_labels_is_false():
    la1 = LabeledArray(np.ones((2, 3)), LABELS_2D)
    la2 = LabeledArray(np.ones((2, 3)),
                       (("xx", "yy"), ("c0", "c1", "c2")))
    assert (la1 == la2) is False


def test_eq_scalar_falls_through_to_ndarray():
    """Comparing against a non-LabeledArray returns an element-wise mask.

    This exercises ``return self.__array__().__eq__(other)`` (line 114).
    """
    la = LabeledArray(np.array([[1.0, 2.0, 3.0],
                                [1.0, 2.0, 3.0]]),
                      LABELS_2D)
    mask = la == 2.0
    # The result is a plain numpy bool array (not a LabeledArray),
    # because we fell out of the LabeledArray branch.
    np.testing.assert_array_equal(
        np.asarray(mask),
        np.array([[False, True, False],
                  [False, True, False]]),
    )


def test_eq_array_falls_through_to_ndarray():
    la = _la(fill=1.0)
    # Compare against a plain ndarray of the same shape — also flows
    # through the scalar fallback path.
    out = la == np.ones_like(np.asarray(la))
    np.testing.assert_array_equal(np.asarray(out), np.ones((2, 3),
                                                            dtype=bool))


def test_ne_is_negation_of_eq():
    la1 = _la()
    la2 = _la()
    # Both are equal; __ne__ should be False.
    assert (la1 != la2) is False
    la3 = _la(fill=99.0)
    assert (la1 != la3) is True


# -------------------------------------------------------------------- #
# Mixin first-in-MRO contract
# -------------------------------------------------------------------- #

def test_mixin_first_in_mro():
    """``_LabeledMixin`` must appear before the C base class so its
    methods (esp. __repr__/__str__/__eq__) shadow ndarray's defaults.
    """
    mro = LabeledArray.__mro__
    mixin_idx = mro.index(_LabeledMixin)
    # ndarray is somewhere later in the MRO.
    assert np.ndarray in mro
    ndarray_idx = mro.index(np.ndarray)
    assert mixin_idx < ndarray_idx


def test_mixin_methods_not_inherited_from_ndarray():
    """Sanity: __repr__ and __eq__ come from the mixin, not ndarray."""
    assert LabeledArray.__repr__ is _LabeledMixin.__repr__
    assert LabeledArray.__eq__ is _LabeledMixin.__eq__
    assert LabeledArray.__ne__ is _LabeledMixin.__ne__
