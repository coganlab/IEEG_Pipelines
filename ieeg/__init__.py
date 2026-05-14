"""Top-level package for ieeg.

Type aliases are split between MNE-free (always available) and MNE-backed
(lazy-imported via PEP 562 module ``__getattr__``). Importing ``ieeg`` no
longer pulls in MNE; ``from ieeg import Signal`` triggers the MNE import
only on first access.
"""
from __future__ import annotations

from os import PathLike as PL
from typing import TYPE_CHECKING

from numpy import ndarray

# --- MNE-free aliases (cheap, always available) ---
PathLike = str | PL
Doubles = tuple[float, float] | list[float, float] | ndarray[(2,), float]
ListNum = int | float | ndarray | list | tuple

# --- MNE-backed aliases (lazy) ---
# Defined here for static type checkers; materialised at runtime by
# ``__getattr__`` below so that ``import ieeg`` does not require MNE.
if TYPE_CHECKING:
    import mne.io
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import base

    Signal = base.BaseRaw | BaseEpochs | Evoked
    RunDict = dict[int, mne.io.Raw]
    SubDict = dict[str, RunDict]


def __getattr__(name: str):
    """PEP 562 lazy attribute resolution for MNE-backed aliases.

    Triggered on ``from ieeg import Signal`` (and friends); resolved
    once and cached on the module ``globals()``.
    """
    if name == "Signal":
        from mne.epochs import BaseEpochs
        from mne.evoked import Evoked
        from mne.io import base
        val = base.BaseRaw | BaseEpochs | Evoked
        globals()[name] = val
        return val
    if name == "RunDict":
        import mne.io
        val = dict[int, mne.io.Raw]
        globals()[name] = val
        return val
    if name == "SubDict":
        # SubDict references RunDict; resolve transitively.
        val = dict[str, __getattr__("RunDict")]
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PathLike",
    "Doubles",
    "ListNum",
    "Signal",
    "RunDict",
    "SubDict",
]
