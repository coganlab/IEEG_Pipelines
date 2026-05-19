"""Coverage for ``ieeg.arrays.__init__`` factory + lazy reexports.

Covers the gaps that ``test_torch_labeled.py`` doesn't reach:

- The ``except ImportError: torch = None`` branch when torch is not
  importable. We can't actually uninstall torch, so we patch
  ``builtins.__import__`` to simulate ``ImportError`` for the duration
  of one call.
- The ``labels is None`` branch of the numpy dispatch.
- The ``AttributeError`` from ``__getattr__`` for unknown names.
"""
from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest

import ieeg.arrays as arrays_pkg


# -------------------------------------------------------------------- #
# Lazy reexports via module __getattr__
# -------------------------------------------------------------------- #

def test_getattr_returns_labeled_array_class():
    from ieeg.arrays.label import LabeledArray as direct
    assert arrays_pkg.LabeledArray is direct


def test_getattr_returns_torch_labeled_array_class():
    pytest.importorskip("torch")
    from ieeg.arrays.torch_labeled import TorchLabeledArray as direct
    assert arrays_pkg.TorchLabeledArray is direct


def test_getattr_unknown_attr_raises():
    with pytest.raises(AttributeError) as excinfo:
        _ = arrays_pkg.DoesNotExist
    msg = str(excinfo.value)
    assert "DoesNotExist" in msg
    assert "ieeg.arrays" in msg


def test_dunder_all_exports():
    expected = {"labeled_array", "LabeledArray", "TorchLabeledArray"}
    assert set(arrays_pkg.__all__) == expected


# -------------------------------------------------------------------- #
# labeled_array() — numpy path branches
# -------------------------------------------------------------------- #

def test_labeled_array_with_none_labels_uses_default_path():
    """``labels=None`` exercises line 60-61 (no-labels constructor)."""
    from ieeg.arrays.label import LabeledArray
    a = np.zeros((2, 3))
    la = arrays_pkg.labeled_array(a)
    assert isinstance(la, LabeledArray)
    assert la.shape == (2, 3)


def test_labeled_array_with_labels_uses_full_constructor():
    from ieeg.arrays.label import LabeledArray
    a = np.zeros((2, 3))
    la = arrays_pkg.labeled_array(a,
                                  labels=(("a", "b"), ("c", "d", "e")))
    assert isinstance(la, LabeledArray)
    assert tuple(la.labels[0]) == ("a", "b")


def test_labeled_array_accepts_list_input():
    from ieeg.arrays.label import LabeledArray
    la = arrays_pkg.labeled_array([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(la, LabeledArray)
    assert la.shape == (2, 2)


# -------------------------------------------------------------------- #
# labeled_array() — simulated torch-missing environment
# -------------------------------------------------------------------- #

@pytest.fixture
def torch_import_blocked(monkeypatch):
    """Make ``import torch`` raise ``ImportError`` for the duration.

    Exercises the ``except ImportError`` branch (lines 52-53) without
    needing an actual torch-less environment.
    """
    real_import = builtins.__import__
    # Drop any cached module so the import statement re-runs the
    # ImportError path.
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def _fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated: torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    yield


def test_factory_falls_back_to_numpy_when_torch_missing(
        torch_import_blocked):
    """When torch can't be imported, the factory should still build
    a numpy ``LabeledArray`` from a plain ndarray."""
    from ieeg.arrays.label import LabeledArray
    la = arrays_pkg.labeled_array(np.zeros((2, 3)))
    assert isinstance(la, LabeledArray)


def test_factory_does_not_call_torch_isinstance_when_missing(
        torch_import_blocked):
    """The numpy path must not crash even though the torch-isinstance
    check was skipped (because torch became None inside the function).
    """
    la = arrays_pkg.labeled_array(np.array([[1.0, 2.0]]),
                                  labels=(("r0",), ("c0", "c1")))
    assert la.shape == (1, 2)
