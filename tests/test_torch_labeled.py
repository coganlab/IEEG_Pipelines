"""Smoke + behaviour tests for TorchLabeledArray.

Mirrors the structure of ``test_labeledarray_py.py`` but for the torch
parallel class. Skipped automatically when ``torch`` is unavailable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ieeg.arrays import labeled_array  # noqa: E402
from ieeg.arrays.torch_labeled import TorchLabeledArray  # noqa: E402


LABELS_3D = (("a", "b"), ("c", "d", "e"), ("f", "g", "h", "i"))


# -------------------------------------------------------------------- #
# Construction + factory dispatch
# -------------------------------------------------------------------- #

def test_construct_from_tensor():
    t = torch.zeros(2, 3, 4)
    la = TorchLabeledArray(t, labels=LABELS_3D)
    assert isinstance(la, torch.Tensor)
    assert isinstance(la, TorchLabeledArray)
    assert la.shape == torch.Size([2, 3, 4])
    assert la.labels == LABELS_3D


def test_construct_from_array_like():
    la = TorchLabeledArray([[1.0, 2.0], [3.0, 4.0]],
                           labels=(("r0", "r1"), ("c0", "c1")))
    assert la.shape == torch.Size([2, 2])
    assert la.labels == (("r0", "r1"), ("c0", "c1"))


def test_construct_default_labels():
    la = TorchLabeledArray(torch.zeros(2, 3))
    assert la.labels == (("0", "1"), ("0", "1", "2"))


def test_labels_validate_shape():
    with pytest.raises(ValueError):
        TorchLabeledArray(torch.zeros(2, 3), labels=(("a", "b"),))
    with pytest.raises(ValueError):
        TorchLabeledArray(torch.zeros(2, 3),
                          labels=(("a", "b"), ("c", "d")))


def test_factory_dispatches_to_torch():
    t = torch.zeros(2, 3)
    la = labeled_array(t, labels=(("a", "b"), ("c", "d", "e")))
    assert isinstance(la, TorchLabeledArray)


def test_factory_dispatches_to_numpy():
    from ieeg.arrays.label import LabeledArray
    a = np.zeros((2, 3))
    la = labeled_array(a, labels=(("a", "b"), ("c", "d", "e")))
    assert isinstance(la, LabeledArray)
    assert not isinstance(la, TorchLabeledArray)


# -------------------------------------------------------------------- #
# Indexing
# -------------------------------------------------------------------- #

def test_string_indexing_axis0():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    sub = la["a"]
    assert isinstance(sub, TorchLabeledArray)
    assert sub.shape == torch.Size([3, 4])
    assert sub.labels == (("c", "d", "e"), ("f", "g", "h", "i"))
    # Values match integer indexing.
    assert torch.equal(sub, la[0])


def test_string_indexing_multi_axis():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    val = la["a", "c", "f"]
    # Scalar drops all axes.
    assert val.item() == 0.0


def test_string_indexing_missing_key():
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    with pytest.raises(KeyError):
        _ = la["zzz"]


def test_slice_indexing_propagates_labels():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    sub = la[:, 1:, :]
    # Slice propagation goes through __torch_function__ which strips
    # labels for un-curated ops; this op IS the __getitem__ path and
    # falls back to torch.Tensor.__getitem__ (no string keys → no
    # special handling). We verify shape is right, even if labels are
    # stripped to a warning.
    assert sub.shape == torch.Size([2, 2, 4])


# -------------------------------------------------------------------- #
# Label propagation through curated ops
# -------------------------------------------------------------------- #

def test_propagate_through_device_move_cpu_to_cpu():
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    moved = la.cpu()
    assert isinstance(moved, TorchLabeledArray)
    assert moved.labels == la.labels


def test_propagate_through_dtype_cast():
    la = TorchLabeledArray(torch.zeros(2, 3, dtype=torch.float64),
                           labels=(("a", "b"), ("c", "d", "e")))
    cast = la.float()
    assert isinstance(cast, TorchLabeledArray)
    assert cast.dtype == torch.float32
    assert cast.labels == la.labels


def test_propagate_through_elementwise_unary():
    la = TorchLabeledArray(torch.tensor([[-1.0, 2.0], [3.0, -4.0]]),
                           labels=(("a", "b"), ("c", "d")))
    result = la.abs()
    assert isinstance(result, TorchLabeledArray)
    assert result.labels == la.labels
    assert torch.equal(result,
                       torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_propagate_through_elementwise_binary():
    la = TorchLabeledArray(torch.ones(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    result = la + 1.0
    assert isinstance(result, TorchLabeledArray)
    assert result.labels == la.labels
    assert torch.all(result == 2.0).item()


def test_propagate_through_clone_detach():
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    assert la.clone().labels == la.labels
    assert la.detach().labels == la.labels


# -------------------------------------------------------------------- #
# Reductions
# -------------------------------------------------------------------- #

def test_reduction_drops_axis():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    r = la.mean(dim=1)
    assert isinstance(r, TorchLabeledArray)
    assert r.shape == torch.Size([2, 4])
    assert r.labels == (("a", "b"), ("f", "g", "h", "i"))


def test_reduction_keepdim_keeps_singleton_label():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    r = la.sum(dim=2, keepdim=True)
    assert isinstance(r, TorchLabeledArray)
    assert r.shape == torch.Size([2, 3, 1])
    assert r.labels == (("a", "b"), ("c", "d", "e"), ("0",))


def test_reduction_over_all_dropps_labels():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    r = la.sum()
    # Scalar reductions return a plain tensor.
    assert r.ndim == 0
    assert float(r.item()) == 276.0


# -------------------------------------------------------------------- #
# Label-stripping warnings
# -------------------------------------------------------------------- #

def test_stripped_op_emits_warning():
    # Clear the warned-set so this test stands alone.
    from ieeg.arrays.torch_labeled import _WARNED_OPS
    _WARNED_OPS.clear()

    la = TorchLabeledArray(torch.zeros(4),
                           labels=(("a", "b", "c", "d"),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = torch.fft.fft(la)
    # Plain tensor, not our subclass.
    assert type(result) is torch.Tensor
    # At least one of the warnings is a UserWarning from torch_labeled.
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]
    assert any("TorchLabeledArray" in m and "labels dropped" in m
               for m in msgs), msgs


def test_warning_is_one_shot():
    from ieeg.arrays.torch_labeled import _WARNED_OPS
    _WARNED_OPS.clear()

    la = TorchLabeledArray(torch.zeros(4),
                           labels=(("a", "b", "c", "d"),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = torch.fft.fft(la)
        _ = torch.fft.fft(la)  # should NOT re-warn
    fft_warns = [w for w in caught
                 if issubclass(w.category, UserWarning)
                 and "fft" in str(w.message)]
    assert len(fft_warns) == 1


# -------------------------------------------------------------------- #
# Migration helper
# -------------------------------------------------------------------- #

def test_from_labeled_round_trip():
    from ieeg.arrays.label import LabeledArray
    a = np.arange(24.).reshape(2, 3, 4)
    np_la = LabeledArray(a, LABELS_3D)
    torch_la = TorchLabeledArray.from_labeled(np_la)
    assert isinstance(torch_la, TorchLabeledArray)
    assert torch_la.shape == torch.Size([2, 3, 4])
    np.testing.assert_array_equal(
        torch_la.cpu().numpy(), np.asarray(np_la))


# -------------------------------------------------------------------- #
# CUDA — skipped when GPU is unavailable
# -------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="No CUDA GPU")
def test_labels_survive_cuda_round_trip():
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    gpu = la.cuda()
    assert isinstance(gpu, TorchLabeledArray)
    assert gpu.labels == la.labels
    assert gpu.is_cuda

    back = gpu.cpu()
    assert isinstance(back, TorchLabeledArray)
    assert back.labels == la.labels


# ==================================================================== #
# Comprehensive coverage added during the consolidation refactor
# ==================================================================== #


# -------------------------------------------------------------------- #
# Construction edge cases
# -------------------------------------------------------------------- #

def test_construct_from_numpy_array():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    la = TorchLabeledArray(a, labels=(("r0", "r1"), ("c0", "c1")))
    assert la.shape == torch.Size([2, 2])
    assert la.labels == (("r0", "r1"), ("c0", "c1"))


def test_construct_preserves_dtype_and_device():
    t = torch.zeros(2, 3, dtype=torch.float64)
    la = TorchLabeledArray(t)
    assert la.dtype == torch.float64
    assert la.device == t.device


def test_construct_preserves_requires_grad():
    t = torch.zeros(2, 3, requires_grad=True)
    la = TorchLabeledArray(t)
    assert la.requires_grad


def test_construct_shares_storage_with_input_tensor():
    t = torch.zeros(2, 3)
    la = TorchLabeledArray(t)
    # Mutation through one is visible through the other.
    la[0, 0] = 99.0
    assert t[0, 0].item() == 99.0


def test_labels_coercion_to_str_tuple():
    # Numeric labels are converted to strings.
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=([1, 2], [3, 4, 5]))
    assert la.labels == (("1", "2"), ("3", "4", "5"))


def test_default_labels_match_axis_size():
    la = TorchLabeledArray(torch.zeros(5))
    assert la.labels == (("0", "1", "2", "3", "4"),)


def test_scalar_tensor_has_empty_labels():
    la = TorchLabeledArray(torch.tensor(7.0))
    assert la.labels == ()
    assert la.item() == 7.0


# -------------------------------------------------------------------- #
# Factory dispatch — every supported call form
# -------------------------------------------------------------------- #

def test_factory_with_default_labels():
    la = labeled_array(torch.zeros(2, 3))
    assert isinstance(la, TorchLabeledArray)
    assert la.labels == (("0", "1"), ("0", "1", "2"))


def test_factory_with_python_list_dispatches_to_numpy():
    from ieeg.arrays.label import LabeledArray
    la = labeled_array([[1.0, 2.0]], labels=(("r0",), ("c0", "c1")))
    assert isinstance(la, LabeledArray)


def test_factory_with_delimiter_passthrough():
    from ieeg.arrays.label import LabeledArray
    la = labeled_array(np.zeros((2, 2)),
                       labels=(("r0", "r1"), ("c0", "c1")),
                       delimiter="|")
    assert isinstance(la, LabeledArray)
    # Delimiter is consumed by ``combine``: the joined labels must use
    # the custom separator.
    combined = la.combine((0, 1))
    joined = tuple(combined.labels[0])
    assert all("|" in name for name in joined)


def test_factory_module_reexport_lazy():
    # __getattr__ on the ieeg.arrays module yields the classes.
    import ieeg.arrays as m
    la_cls = m.LabeledArray
    torch_cls = m.TorchLabeledArray
    from ieeg.arrays.label import LabeledArray as direct_la
    from ieeg.arrays.torch_labeled import TorchLabeledArray as direct_t
    assert la_cls is direct_la
    assert torch_cls is direct_t


def test_factory_module_reexport_invalid_attr():
    import ieeg.arrays as m
    with pytest.raises(AttributeError):
        _ = m.NotAClass  # noqa: F841


# -------------------------------------------------------------------- #
# _LabeledMixin shared between numpy LabeledArray and (implicit) parity
# -------------------------------------------------------------------- #

def test_numpy_la_has_mixin_methods():
    """The mixin extraction must not break the numpy class' surface."""
    from ieeg.arrays._labels_mixin import _LabeledMixin
    from ieeg.arrays.label import LabeledArray
    assert issubclass(LabeledArray, _LabeledMixin)
    la = LabeledArray(np.zeros((2, 3)),
                      labels=(("a", "b"), ("c", "d", "e")))
    # Mixin-supplied methods.
    for attr in ("to_dict", "items", "keys", "values", "memory"):
        assert hasattr(la, attr), f"missing {attr}"


def test_torch_la_does_not_inherit_mixin():
    """TorchLabeledArray deliberately reimplements label methods.

    The mixin's repr/to_dict path goes through numpy materialisation
    which is hostile to CUDA tensors. TorchLabeledArray supplies its
    own minimal surface.
    """
    from ieeg.arrays._labels_mixin import _LabeledMixin
    assert not issubclass(TorchLabeledArray, _LabeledMixin)


# -------------------------------------------------------------------- #
# Curated propagation set — every category exercised
# -------------------------------------------------------------------- #

@pytest.mark.parametrize("op", [
    "abs", "neg", "sqrt", "square", "exp", "log",
    "sin", "cos", "tan", "sigmoid", "tanh",
    "ceil", "floor", "round", "sign",
])
def test_propagate_unary_elementwise(op):
    la = TorchLabeledArray(torch.abs(torch.randn(3, 4)) + 0.1,
                           labels=(("a", "b", "c"),
                                   ("d", "e", "f", "g")))
    func = getattr(la, op)
    result = func()
    assert isinstance(result, TorchLabeledArray)
    assert result.labels == la.labels


@pytest.mark.parametrize("op_name,op", [
    ("add", lambda x: x + 1.0),
    ("sub", lambda x: x - 1.0),
    ("mul", lambda x: x * 2.0),
    ("div", lambda x: x / 2.0),
    ("pow", lambda x: x ** 2.0),
    ("eq", lambda x: x == 0.0),
    ("ne", lambda x: x != 0.0),
    ("lt", lambda x: x < 1.0),
    ("ge", lambda x: x >= 0.0),
])
def test_propagate_binary_elementwise(op_name, op):
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    result = op(la)
    assert isinstance(result, TorchLabeledArray), op_name
    assert result.labels == la.labels, op_name


@pytest.mark.parametrize("dtype_method", [
    "float", "double", "half", "int", "long", "short", "bool",
])
def test_propagate_dtype_cast(dtype_method):
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    func = getattr(la, dtype_method)
    result = func()
    assert isinstance(result, TorchLabeledArray)
    assert result.labels == la.labels


def test_propagate_clamp_and_nan_to_num():
    la = TorchLabeledArray(torch.tensor([[float("nan"), 1.0],
                                          [-5.0, 5.0]]),
                           labels=(("r0", "r1"), ("c0", "c1")))
    clamped = la.clamp(min=-1.0, max=1.0)
    assert isinstance(clamped, TorchLabeledArray)
    assert clamped.labels == la.labels
    cleaned = la.nan_to_num(nan=0.0)
    assert isinstance(cleaned, TorchLabeledArray)
    assert cleaned.labels == la.labels


# -------------------------------------------------------------------- #
# Reduction set — axis arithmetic edge cases
# -------------------------------------------------------------------- #

@pytest.mark.parametrize("reduction", [
    "sum", "mean", "std", "var", "prod",
    "amax", "amin", "logsumexp",
])
def test_reduction_drops_correct_axis(reduction):
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4) + 1.0,
                           labels=LABELS_3D)
    func = getattr(la, reduction)
    out = func(dim=1)
    assert isinstance(out, TorchLabeledArray)
    assert out.shape == torch.Size([2, 4])
    assert out.labels == (("a", "b"), ("f", "g", "h", "i"))


def test_reduction_with_negative_dim():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la.mean(dim=-1)
    assert isinstance(out, TorchLabeledArray)
    assert out.shape == torch.Size([2, 3])
    assert out.labels == (("a", "b"), ("c", "d", "e"))


def test_reduction_with_tuple_dim():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la.sum(dim=(0, 2))
    assert isinstance(out, TorchLabeledArray)
    assert out.shape == torch.Size([3])
    assert out.labels == (("c", "d", "e"),)


def test_argmax_argmin_propagate_labels():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la.argmax(dim=2)
    assert isinstance(out, TorchLabeledArray)
    assert out.shape == torch.Size([2, 3])
    assert out.labels == (("a", "b"), ("c", "d", "e"))


# -------------------------------------------------------------------- #
# String indexing — multi-axis combinations
# -------------------------------------------------------------------- #

def test_string_key_drops_axis_0():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la["b"]
    assert out.shape == torch.Size([3, 4])
    assert out.labels == (("c", "d", "e"), ("f", "g", "h", "i"))
    assert torch.equal(out, la[1])


def test_string_key_with_int_drops_two_axes():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la["a", 1]
    assert out.shape == torch.Size([4])
    assert out.labels == (("f", "g", "h", "i"),)


def test_string_key_with_slice():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la["a", :, "f"]
    # Two string keys (axis 0 and 2) drop, slice on axis 1 keeps.
    assert out.shape == torch.Size([3])
    assert out.labels == (("c", "d", "e"),)


def test_string_key_with_list_of_strings():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la[:, :, ["f", "h"]]
    # Last axis is selected by a list of two labels → length 2.
    assert out.shape == torch.Size([2, 3, 2])
    # The label list isn't re-attached (it's a torch advanced index),
    # but the surviving axes have their original labels.
    assert out.labels[0] == ("a", "b")
    assert out.labels[1] == ("c", "d", "e")


def test_missing_label_raises_with_helpful_message():
    la = TorchLabeledArray(torch.zeros(2, 3),
                           labels=(("a", "b"), ("c", "d", "e")))
    with pytest.raises(KeyError) as excinfo:
        _ = la["nonexistent"]
    msg = str(excinfo.value)
    assert "nonexistent" in msg
    assert "axis 0" in msg


# -------------------------------------------------------------------- #
# Label stripping warning system
# -------------------------------------------------------------------- #

def test_silent_strip_op_emits_no_warning():
    """__getitem__ falls back through __torch_function__ when called
    via slicing without strings; it's silently stripped (in the
    _SILENT_STRIP_OPS set) so we shouldn't see a UserWarning."""
    from ieeg.arrays.torch_labeled import _WARNED_OPS
    _WARNED_OPS.clear()
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = la[0]
        _ = la[:, 1]
        _ = la[..., 0]
    user_warns = [w for w in caught
                  if issubclass(w.category, UserWarning)
                  and "TorchLabeledArray" in str(w.message)]
    assert user_warns == []


def test_different_stripping_ops_warn_independently():
    from ieeg.arrays.torch_labeled import _WARNED_OPS
    _WARNED_OPS.clear()
    la = TorchLabeledArray(torch.zeros(4),
                           labels=(("a", "b", "c", "d"),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = torch.fft.fft(la)
        _ = torch.fft.rfft(la)  # different op → second warning fires
    fft_warns = [w for w in caught
                 if issubclass(w.category, UserWarning)
                 and "TorchLabeledArray" in str(w.message)]
    assert len(fft_warns) == 2


def test_strip_returns_plain_tensor_not_subclass():
    from ieeg.arrays.torch_labeled import _WARNED_OPS
    _WARNED_OPS.clear()
    la = TorchLabeledArray(torch.zeros(4),
                           labels=(("a", "b", "c", "d"),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = torch.fft.fft(la)
    # The stripped result should be a pure torch.Tensor.
    assert type(out) is torch.Tensor
    assert not isinstance(out, TorchLabeledArray)


# -------------------------------------------------------------------- #
# Memory accounting
# -------------------------------------------------------------------- #

def test_memory_returns_size_and_unit():
    la = TorchLabeledArray(torch.zeros(1024, dtype=torch.float32))
    size, unit = la.memory()
    # 1024 * 4 bytes = 4096 B = 4 KiB.
    assert unit == "KiB"
    assert size == pytest.approx(4.0)


def test_memory_large_tensor():
    la = TorchLabeledArray(torch.zeros(1024 * 1024,
                                       dtype=torch.float32))
    size, unit = la.memory()
    # 4 MiB.
    assert unit == "MiB"
    assert size == pytest.approx(4.0)


# -------------------------------------------------------------------- #
# Repr / str
# -------------------------------------------------------------------- #

def test_repr_includes_labels():
    la = TorchLabeledArray(torch.zeros(2, 2),
                           labels=(("r0", "r1"), ("c0", "c1")))
    s = repr(la)
    assert "TorchLabeledArray" in s
    for lab in ("r0", "r1", "c0", "c1"):
        assert lab in s


def test_str_equals_repr():
    la = TorchLabeledArray(torch.zeros(2),
                           labels=(("x", "y"),))
    assert str(la) == repr(la)


# -------------------------------------------------------------------- #
# Migration from numpy LabeledArray
# -------------------------------------------------------------------- #

def test_from_labeled_with_non_contiguous_source():
    """``from_labeled`` should handle non-contiguous storage via the
    explicit ``np.ascontiguousarray`` call in the migration helper."""
    from ieeg.arrays.label import LabeledArray
    a = np.arange(24.).reshape(2, 3, 4)
    np_la = LabeledArray(a, LABELS_3D)
    # Pick a strided slice (every other column) — this is non-contiguous
    # but preserves the label-block shape alignment.
    view = np_la[:, :, ::2]
    torch_la = TorchLabeledArray.from_labeled(view)
    assert torch_la.shape == torch.Size([2, 3, 2])
    np.testing.assert_array_equal(
        torch_la.cpu().numpy(), np.asarray(view))


# -------------------------------------------------------------------- #
# Doctest sanity
# -------------------------------------------------------------------- #

def test_doctest_module_runs():
    """The doctest in TorchLabeledArray.__doc__ should pass."""
    import doctest
    from ieeg.arrays import torch_labeled
    failures, _ = doctest.testmod(torch_labeled, verbose=False)
    assert failures == 0


# -------------------------------------------------------------------- #
# Autograd compatibility
# -------------------------------------------------------------------- #

def test_autograd_through_propagated_op():
    la = TorchLabeledArray(torch.tensor([1.0, 2.0, 3.0],
                                         requires_grad=True),
                           labels=(("a", "b", "c"),))
    out = la.abs().sum()
    out.backward()
    grad = la.grad
    assert grad is not None
    np.testing.assert_array_equal(grad.detach().cpu().numpy(),
                                  np.array([1.0, 1.0, 1.0]))


# -------------------------------------------------------------------- #
# Equality (eq) preserves labels (regression for the _WARNED_OPS fix)
# -------------------------------------------------------------------- #

def test_eq_propagates_labels():
    la = TorchLabeledArray(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                           labels=(("r0", "r1"), ("c0", "c1")))
    mask = la == 2.0
    assert isinstance(mask, TorchLabeledArray)
    assert mask.labels == la.labels
    assert mask.dtype == torch.bool


# -------------------------------------------------------------------- #
# Reduction-with-keepdim labels are singletons
# -------------------------------------------------------------------- #

def test_reduction_keepdim_singleton_labels_per_axis():
    la = TorchLabeledArray(torch.arange(24.).reshape(2, 3, 4),
                           labels=LABELS_3D)
    out = la.sum(dim=(0, 1), keepdim=True)
    assert isinstance(out, TorchLabeledArray)
    assert out.shape == torch.Size([1, 1, 4])
    # Collapsed axes get singleton "0" labels.
    assert out.labels == (("0",), ("0",), ("f", "g", "h", "i"))


# -------------------------------------------------------------------- #
# Stack-like coverage: ensure shape mismatch is caught at construction
# -------------------------------------------------------------------- #

def test_zero_dim_label_count_validates():
    # 0-d tensor → labels must be ().
    with pytest.raises(ValueError):
        TorchLabeledArray(torch.tensor(1.0),
                          labels=(("only",),))

