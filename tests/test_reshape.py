"""Coverage for ``ieeg.arrays.reshape``: stitch/merge/pad/sliding-window
helpers.

The module already has rich doctests for the happy path; this file
targets the **error branches** and **non-numpy backends** that
doctests don't cover.
"""
from __future__ import annotations

import numpy as np
import pytest

from ieeg.arrays.reshape import (
    stitch_mats,
    merge,
    make_data_same,
    pad_to_match,
    rand_offset_reshape,
    sliding_window_view,
)


# -------------------------------------------------------------------- #
# stitch_mats — error branch
# -------------------------------------------------------------------- #

def test_stitch_mats_mismatched_overlap_count():
    """``len(overlaps) != len(mats) - 1`` must raise ValueError."""
    m1 = np.array([[1, 2], [3, 4]])
    m2 = np.array([[5, 6], [7, 8]])
    m3 = np.array([[9, 10], [11, 12]])
    # Three matrices need exactly two overlaps; supply one.
    with pytest.raises(ValueError, match="number of overlaps"):
        stitch_mats([m1, m2, m3], [1])


def test_stitch_mats_falls_back_to_float_when_nan_present():
    """When the blended values include NaN, the integer cast yields
    sentinel values that do not equal the original — exercises the
    AssertionError branch (line 94-95)."""
    m1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    m2 = np.array([[4.0, 5.0, float("nan")]], dtype=np.float64)
    out = stitch_mats([m1, m2], [0], axis=1)
    # Float dtype preserved (NaN cannot survive an integer cast).
    assert out.dtype == np.float64
    assert np.isnan(out).any()


# -------------------------------------------------------------------- #
# merge — basic shape contract
# -------------------------------------------------------------------- #

def test_merge_returns_three_chunks():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(12, 24, dtype=np.float64).reshape(3, 4)
    out = merge(a, b, overlap=2, axis=1)
    assert isinstance(out, list)
    assert len(out) == 3
    # Total width = (4 - 2) + 2 + (4 - 2) = 6 along the merge axis.
    total = sum(chunk.shape[1] for chunk in out)
    assert total == 6


def test_merge_overlap_zero_yields_empty_middle():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
    out = merge(a, b, overlap=0, axis=1)
    # Middle slice is empty (zero-width).
    assert out[1].shape[1] == 0


# -------------------------------------------------------------------- #
# pad_to_match — scalar axis handling, multi-axis padding
# -------------------------------------------------------------------- #

def test_pad_to_match_scalar_axis_is_promoted_to_tuple():
    big = np.zeros((4, 6))
    small = np.zeros((4, 4))
    out = pad_to_match(big, small, axis=0)
    assert out.shape[1] == 6


def test_pad_to_match_returns_input_when_shapes_already_match():
    big = np.zeros((3, 4))
    small = np.zeros((3, 4))
    out = pad_to_match(big, small)
    # No padding needed → returns sig2 unmodified.
    assert out is small


def test_pad_to_match_negative_axis():
    big = np.zeros((3, 8))
    small = np.zeros((3, 5))
    # axis=-1 should be normalised so the last axis is preserved
    # and the others (none) pad.
    out = pad_to_match(big, small, axis=-1)
    assert out.shape == small.shape  # nothing to pad


# -------------------------------------------------------------------- #
# rand_offset_reshape — edge: divisible / non-divisible
# -------------------------------------------------------------------- #

def test_rand_offset_reshape_when_divisible():
    """When ``data.shape[pad_ax] % shape[pad_ax] == 0`` the function
    decrements num_stack — exercises line 324."""
    data = np.arange(20, dtype=np.float64).reshape(2, 10)
    out = rand_offset_reshape(data, (2, 5), stack_ax=0, pad_ax=1, rng=0)
    # 10 // 5 = 2, exactly divisible → num_stack -= 1 → 1 stack only.
    # Output stacks = 2 * 1 = 2 rows.
    assert out.shape == (2, 5)


def test_rand_offset_reshape_when_not_divisible():
    data = np.arange(22, dtype=np.float64).reshape(2, 11)
    out = rand_offset_reshape(data, (2, 5), stack_ax=0, pad_ax=1, rng=0)
    # 11 // 5 = 2, not divisible → num_stack = 2 → output stacks = 4.
    assert out.shape == (4, 5)


def test_rand_offset_reshape_rng_int_seed_is_deterministic():
    data = np.arange(50, dtype=np.float64).reshape(5, 10)
    a = rand_offset_reshape(data, (2, 4), 0, 1, rng=42)
    b = rand_offset_reshape(data, (2, 4), 0, 1, rng=42)
    np.testing.assert_array_equal(a, b)


# -------------------------------------------------------------------- #
# make_data_same — both branches (pad and rand_offset)
# -------------------------------------------------------------------- #

def test_make_data_same_pads_when_smaller():
    """``data.shape[pad_ax] <= shape[pad_ax]`` → pad path."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    out = make_data_same(data, (2, 6), make_stacks_same=False)
    assert out.shape == (2, 6)


def test_make_data_same_subsets_when_larger():
    """``data.shape[pad_ax] > shape[pad_ax]`` → rand_offset path."""
    data = np.arange(20).reshape(2, 10)
    out = make_data_same(data, (2, 4), make_stacks_same=False, rng=0)
    assert out.shape[1] == 4
    # Stack axis was doubled by rand_offset (num_stack=2).
    assert out.shape[0] >= 2


def test_make_data_same_subsamples_stacks_when_too_many():
    """``out.shape[stack_ax] > shape[stack_ax]`` AND
    ``make_stacks_same=True`` → np.take with rng.choice (line 189-191)."""
    data = np.arange(60).reshape(2, 30)
    out = make_data_same(data, (2, 5), make_stacks_same=True, rng=0)
    assert out.shape == (2, 5)


def test_make_data_same_oversamples_stacks_when_too_few():
    """``out.shape[stack_ax] < shape[stack_ax]`` AND
    ``make_stacks_same=True`` → concat with rng.choice (line 192-196)."""
    data = np.arange(8).reshape(2, 4)
    # Match a stack axis of 8 (oversample from 2 to 8).
    out = make_data_same(data, (8, 4), stack_ax=0, pad_ax=1,
                         make_stacks_same=True, rng=0)
    assert out.shape == (8, 4)


# -------------------------------------------------------------------- #
# sliding_window_view — error and edge paths
# -------------------------------------------------------------------- #

def test_sliding_window_view_negative_shape_raises():
    x = np.arange(10)
    with pytest.raises(ValueError, match="cannot contain negative"):
        sliding_window_view(x, -3)


def test_sliding_window_view_shape_axis_mismatch_raises():
    x = np.arange(12).reshape(3, 4)
    # Two-axis array, but window_shape only has 1 element and axis=None.
    with pytest.raises(ValueError, match="window_shape for all dimensions"):
        sliding_window_view(x, 2)


def test_sliding_window_view_shape_axis_count_mismatch():
    x = np.arange(12).reshape(3, 4)
    # window_shape and axis disagree in length.
    with pytest.raises(ValueError, match="matching length"):
        sliding_window_view(x, (2, 2), axis=0)


def test_sliding_window_view_window_too_large_raises():
    x = np.arange(5)
    with pytest.raises(ValueError, match="cannot be larger"):
        sliding_window_view(x, 10)


def test_sliding_window_view_torch_backend():
    """torch path goes through ``xp.as_strided`` with ``stride()``."""
    torch = pytest.importorskip("torch")
    x = torch.arange(6, dtype=torch.float32)
    v = sliding_window_view(x, 3)
    assert v.shape == (4, 3)
    # First window is [0, 1, 2].
    assert torch.equal(v[0], torch.tensor([0.0, 1.0, 2.0]))


def test_sliding_window_view_torch_two_axis():
    torch = pytest.importorskip("torch")
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    v = sliding_window_view(x, (2, 2))
    assert v.shape == (2, 3, 2, 2)


# -------------------------------------------------------------------- #
# normalize_axis_tuple polyfill — the fallback definition is exercised
# implicitly when running on numpy<2.0 environments. Without that, the
# import succeeds. We just verify the symbol is in scope and behaves
# correctly through sliding_window_view's call.
# -------------------------------------------------------------------- #

def test_normalize_axis_tuple_negative_axis_handled():
    x = np.arange(12).reshape(3, 4)
    v = sliding_window_view(x, 2, axis=-1)
    assert v.shape == (3, 3, 2)


def test_normalize_axis_tuple_repeated_axis_allowed():
    """``allow_duplicate=True`` is used by sliding_window_view to
    permit the same axis multiple times."""
    x = np.arange(12).reshape(3, 4)
    # window_shape (2, 3) on axis (1, 1) — same axis used twice.
    v = sliding_window_view(x, (2, 3), axis=(1, 1))
    # Both window operations reduce axis 1, so the result is shaped
    # (3, 1, 2, 3) per the doctest.
    assert v.shape == (3, 1, 2, 3)
