"""Coverage for ``ieeg.process`` validators and helpers.

Focuses on testable validators (``ensure_int``, ``validate_type``,
``is_number``) and the COLA constructor's input validation. Skips the
joblib-heavy functions (``parallelize``, ``proc_array``,
``sliding_window``) which already have happy-path doctests and would
need substantial fixture support for new tests.
"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
import pytest

from ieeg.process import (
    ensure_int,
    validate_type,
    is_number,
    iterate_axes,
    COLA,
    _check_store,
    _check_cola,
    _Storer,
    get_mem,
)


# -------------------------------------------------------------------- #
# ensure_int
# -------------------------------------------------------------------- #

def test_ensure_int_accepts_python_int():
    assert ensure_int(5) == 5


def test_ensure_int_accepts_numpy_int():
    assert ensure_int(np.int64(7)) == 7


def test_ensure_int_rejects_float():
    with pytest.raises(TypeError, match="must be an int"):
        ensure_int(1.0)


def test_ensure_int_rejects_string():
    with pytest.raises(TypeError, match="must be an int"):
        ensure_int("1")


def test_ensure_int_rejects_bool():
    """``bool`` is an ``int`` subclass in Python but should be rejected
    here per the docstring rationale."""
    with pytest.raises(TypeError):
        ensure_int(True)


def test_ensure_int_uses_name_in_error_message():
    with pytest.raises(TypeError, match="my_param"):
        ensure_int(1.5, name="my_param")


def test_ensure_int_uses_extra_in_error_message():
    with pytest.raises(TypeError, match="something specific"):
        ensure_int(1.5, extra="something specific")


# -------------------------------------------------------------------- #
# validate_type
# -------------------------------------------------------------------- #

def test_validate_type_int_passthrough():
    """``validate_type(x, int)`` calls ensure_int and tolerates real ints."""
    validate_type(5, int)  # should not raise


def test_validate_type_float_uses_is_number():
    """``validate_type(x, float)`` is satisfied by anything ``is_number``."""
    validate_type(1.5, float)
    validate_type(2, float)  # int passes via is_number too


def test_validate_type_basic_isinstance_check():
    validate_type([1, 2, 3], list)


def test_validate_type_rejects_wrong_type():
    with pytest.raises(TypeError, match="must be an instance of"):
        validate_type("a string", list)


def test_validate_type_typevar_constraints():
    T = TypeVar("T", int, float)
    validate_type(1, T)  # int satisfies the TypeVar
    validate_type(2.0, T)  # float satisfies it
    with pytest.raises(TypeError):
        validate_type("not numeric", T)


def test_validate_type_returns_none_when_valid():
    # Function returns None on success.
    assert validate_type(1, int) is None


# -------------------------------------------------------------------- #
# is_number
# -------------------------------------------------------------------- #

def test_is_number_python_int():
    assert is_number(5) is True


def test_is_number_python_float():
    assert is_number(1.5) is True


def test_is_number_numpy_scalar():
    assert is_number(np.float32(3.14)) is True


def test_is_number_numeric_string():
    assert is_number("3.14") is True
    assert is_number("42") is True


def test_is_number_non_numeric_string():
    assert is_number("hello") is False


def test_is_number_arbitrary_object():
    assert is_number([1, 2, 3]) is False
    assert is_number({"a": 1}) is False
    assert is_number(None) is False


def test_is_number_pandas_dataframe_numeric():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert is_number(df) is True


def test_is_number_pandas_dataframe_non_numeric():
    df = pd.DataFrame({"x": ["foo", "bar"]})
    assert is_number(df) is False


def test_is_number_pandas_series_numeric():
    s = pd.Series([1.0, 2.0, 3.0])
    assert is_number(s) is True


def test_is_number_pandas_series_non_numeric():
    s = pd.Series(["foo", "bar"])
    assert is_number(s) is False


# -------------------------------------------------------------------- #
# iterate_axes — additional shapes (the failing doctest stays untouched)
# -------------------------------------------------------------------- #

def test_iterate_axes_single_axis():
    arr = np.arange(6).reshape(2, 3)
    slices = list(iterate_axes(arr, (0,)))
    # Iterating over axis 0 (size 2) yields exactly 2 slice tuples.
    assert len(slices) == 2
    np.testing.assert_array_equal(arr[slices[0]], np.array([0, 1, 2]))
    np.testing.assert_array_equal(arr[slices[1]], np.array([3, 4, 5]))


def test_iterate_axes_count_matches_product_of_shape():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    slices = list(iterate_axes(arr, (0, 1, 2)))
    # All combinations of (2, 3, 4) → 24 scalar slices.
    assert len(slices) == 24


def test_iterate_axes_empty_axes_tuple_yields_single_iteration():
    arr = np.arange(6).reshape(2, 3)
    slices = list(iterate_axes(arr, ()))
    assert len(slices) == 1
    # Every axis is a full slice.
    assert all(s == slice(None) for s in slices[0])


# -------------------------------------------------------------------- #
# COLA constructor validation
# -------------------------------------------------------------------- #

def _noop(x):
    return (x,)


def test_cola_rejects_non_positive_n_samples():
    out = np.zeros(10)
    with pytest.raises(ValueError, match="n_samples must be > 0"):
        COLA(_noop, out, n_total=20, n_samples=0, n_overlap=0, sfreq=100.)


def test_cola_rejects_negative_n_overlap():
    out = np.zeros(10)
    with pytest.raises(ValueError, match="n_overlap must be >= 0"):
        COLA(_noop, out, n_total=20, n_samples=10, n_overlap=-1, sfreq=100.)


def test_cola_rejects_negative_n_total():
    out = np.zeros(10)
    with pytest.raises(ValueError, match="n_total must be >= 0"):
        COLA(_noop, out, n_total=-5, n_samples=10, n_overlap=0, sfreq=100.)


def test_cola_rejects_n_total_smaller_than_window():
    out = np.zeros(10)
    with pytest.raises(ValueError, match="Number of samples per window"):
        COLA(_noop, out, n_total=5, n_samples=10, n_overlap=0, sfreq=100.)


def test_cola_rejects_non_callable_process():
    out = np.zeros(10)
    with pytest.raises(TypeError, match="process must be callable"):
        COLA("not callable", out, n_total=20, n_samples=10, n_overlap=5,
             sfreq=100.)


def test_cola_constructs_with_valid_args():
    out = np.zeros(20)
    cola = COLA(_noop, out, n_total=20, n_samples=10, n_overlap=5, sfreq=100.)
    # Internal state initialised correctly.
    assert cola._n_samples == 10
    assert cola._n_overlap == 5
    assert cola._step == 5
    assert cola._idx == 0


# -------------------------------------------------------------------- #
# _check_store
# -------------------------------------------------------------------- #

def test_check_store_accepts_callable():
    def store(x):
        pass
    assert _check_store(store) is store


def test_check_store_promotes_single_ndarray():
    out = np.zeros(10)
    result = _check_store(out)
    # The ndarray is wrapped in a _Storer (callable).
    assert callable(result)


def test_check_store_promotes_list_of_ndarrays():
    outs = [np.zeros(10), np.zeros(10)]
    result = _check_store(outs)
    assert callable(result)


def test_check_store_rejects_non_callable_non_array():
    with pytest.raises(TypeError, match="store must be callable"):
        _check_store("not valid")


# -------------------------------------------------------------------- #
# _check_cola — direct call
# -------------------------------------------------------------------- #

def test_check_cola_returns_constant_for_valid_window():
    from scipy.signal import get_window
    nperseg = 10
    step = 5
    win = get_window("hann", nperseg, fftbins=True)
    const = _check_cola(win, nperseg, step, "hann")
    # For a valid Hann window with 50% overlap the binsum is constant.
    assert const > 0


def test_check_cola_raises_for_bad_window():
    """Use a non-COLA-compliant window/step combination."""
    win = np.ones(10)  # boxcar
    # Step=3 with boxcar of length 10 → non-constant sum.
    with pytest.raises(ValueError, match="does not provide a constant"):
        _check_cola(win, 10, 3, "boxcar")


# -------------------------------------------------------------------- #
# _Storer
# -------------------------------------------------------------------- #

def test_storer_call_writes_chunks_sequentially():
    out = np.zeros(20)
    storer = _Storer(out)
    storer(np.ones(5))
    storer(np.full(5, 2.0))
    np.testing.assert_array_equal(out[:5], np.ones(5))
    np.testing.assert_array_equal(out[5:10], np.full(5, 2.0))
    np.testing.assert_array_equal(out[10:], np.zeros(10))


def test_storer_rejects_mismatched_outs_count():
    out1 = np.zeros(20)
    out2 = np.zeros(20)
    storer = _Storer(out1, out2)
    # Provide one output when two were expected.
    with pytest.raises(ValueError, match="Bad outs"):
        storer(np.ones(5))


def test_storer_rejects_non_array_in_init():
    with pytest.raises(TypeError, match="must be >= 1D ndarray"):
        _Storer("not an array")


# -------------------------------------------------------------------- #
# get_mem — smoke (depends on psutil, which is an optional dep)
# -------------------------------------------------------------------- #

def test_get_mem_returns_positive_value():
    psutil = pytest.importorskip("psutil")
    mem = get_mem()
    assert mem > 0


def test_get_mem_with_n_jobs_partitions():
    psutil = pytest.importorskip("psutil")
    full = get_mem()
    half = get_mem(n_jobs=1)
    assert 0 < half <= full
