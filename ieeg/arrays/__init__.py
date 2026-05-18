"""Labeled array containers.

This package provides two parallel labeled-array implementations that
share the same per-axis label semantics:

* :class:`ieeg.arrays.label.LabeledArray` — a ``numpy.ndarray`` subclass
  backed by hand-tuned C kernels. This is the canonical, popular path
  used across the pipeline.
* :class:`ieeg.arrays.torch_labeled.TorchLabeledArray` — a
  ``torch.Tensor`` subclass that propagates labels through a curated
  set of operations via ``__torch_function__``. Imported lazily so
  ``torch`` stays an optional dependency.

For most code, prefer the :func:`labeled_array` factory below: it
inspects the input and dispatches to the right subclass. Existing
``LabeledArray(np_array, labels)`` call sites continue to work
unchanged.
"""
from __future__ import annotations

from typing import Any


def labeled_array(data: Any, labels=None, delimiter: str = "-"):
    """Construct the right LabeledArray subclass for the input array type.

    Parameters
    ----------
    data : array_like or torch.Tensor
        The underlying numeric storage. If a ``torch.Tensor``, the
        result is a :class:`TorchLabeledArray`; otherwise it is the
        numpy-backed :class:`LabeledArray`.
    labels : sequence of sequence of str, optional
        Per-axis labels. Forwarded to the chosen subclass.
    delimiter : str, optional
        Delimiter used when combining labels in the numpy path.
        Ignored for the torch path (which has no delimiter notion).

    Returns
    -------
    LabeledArray or TorchLabeledArray
        The appropriate subclass for the input.

    Notes
    -----
    ``torch`` is imported lazily, so this function works in
    environments where the optional ``torch`` extra is not installed
    (the numpy path is always available).
    """
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(data, torch.Tensor):
        from .torch_labeled import TorchLabeledArray
        return TorchLabeledArray(data, labels=labels)

    from .label import LabeledArray
    if labels is None:
        return LabeledArray(data, delimiter=delimiter)
    return LabeledArray(data, labels, delimiter=delimiter)


def __getattr__(name: str):
    # Lazy re-exports so importing ``ieeg.arrays`` stays cheap.
    if name == "LabeledArray":
        from .label import LabeledArray
        return LabeledArray
    if name == "TorchLabeledArray":
        from .torch_labeled import TorchLabeledArray
        return TorchLabeledArray
    raise AttributeError(f"module 'ieeg.arrays' has no attribute {name!r}")


__all__ = ["labeled_array", "LabeledArray", "TorchLabeledArray"]
