"""Backend-agnostic label management for LabeledArray subclasses.

This mixin holds the pure-Python ergonomics that both the NumPy-backed
:class:`ieeg.arrays.label.LabeledArray` and the (future) torch-backed
:class:`ieeg.arrays.torch_labeled.TorchLabeledArray` share. Anything that
depends on a specific array library (np.isnan, np.array_equal, etc.) is
either guarded with array-API dispatch or left in the subclass.

The mixin is intended to be placed **first** in the MRO so its methods
shadow whatever the array library's base class (`ndarray`, `torch.Tensor`)
provides for the same name:

    class LabeledArray(_LabeledMixin, cla):  # cla = C base from labeledarray.c
        ...

    class TorchLabeledArray(_LabeledMixin, torch.Tensor):
        ...
"""
from __future__ import annotations

import numpy as np


class _LabeledMixin:
    """Pure-Python label management shared by all LabeledArray flavours.

    Assumes the subclass exposes:
        - ``self.labels``: sequence of per-axis label sequences
        - ``self.__array__()``: returns a buffer compatible with the
          underlying array library's repr (NumPy ndarray; for torch this
          is the CPU view via ``torch.Tensor.__array__``)
        - ``self.nbytes`` *or* ``.numel() + .element_size()`` for memory
          accounting
    """

    # --- Display ----------------------------------------------------------

    def __repr__(self):
        return repr(self.__array__()) + f"\nlabels({self._label_formatter()})"

    def __str__(self):
        return str(self.__array__()) + f"\nlabels({self._label_formatter()})"

    def _label_formatter(self):
        # Local import to avoid circular import at module load
        from ieeg.arrays.label import Labels

        def _liststr(x):
            return f"\n       ".join(x)

        return _liststr([str(Labels(lab)) for lab in self.labels])

    # --- Memory accounting -----------------------------------------------

    def memory(self):
        """Return ``(size, unit)`` for the underlying buffer.

        Works for both numpy ndarrays (``nbytes``) and torch tensors
        (``numel() * element_size()``).
        """
        if hasattr(self, "nbytes"):
            size = self.nbytes
        elif hasattr(self, "numel") and hasattr(self, "element_size"):
            size = self.numel() * self.element_size()
        else:
            raise TypeError(
                f"Cannot compute memory size for {type(self).__name__}: "
                "neither .nbytes nor (.numel + .element_size) available"
            )
        for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
            if size < 1024.0 or unit == "PiB":
                break
            size /= 1024.0
        return size, unit

    # --- Dict API ---------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a nested dictionary keyed by labels.

        Entries whose value is all-NaN at the leaves are skipped.
        Uses ``np.isnan`` on the leaf; for torch tensors this triggers
        the ``__array__`` round-trip through CPU.
        """
        out = {}
        for k, v in self.items():
            if len(self.labels) > 1:
                out[k] = v.to_dict()
            elif np.isnan(np.asarray(v)).all():
                continue
            else:
                out[k] = v
        return out

    def items(self):
        return zip(self.keys(), self.values())

    def keys(self):
        return (lab for lab in self.labels[0])

    def values(self):
        return (a for a in self)

    # --- Equality ---------------------------------------------------------

    def __eq__(self, other):
        # Local import to avoid circular reference at module-load time
        from ieeg.arrays.label import LabeledArray

        if isinstance(other, LabeledArray):
            return np.array_equal(np.asarray(self), np.asarray(other), True) and \
                all(np.array_equal(np.asarray(l1), np.asarray(l2))
                    for l1, l2 in zip(self.labels, other.labels))
        return self.__array__().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)
