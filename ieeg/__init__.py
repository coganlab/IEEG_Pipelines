"""ieeg: a package for working with intracranial EEG data

.. include:: ../ieeg/README.md
    :parser: myst_parser.sphinx_
"""
from os import PathLike as PL

import mne.io
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base
from numpy import ndarray, random
from numba import njit


PathLike = str | PL
RunDict = dict[int, mne.io.Raw]
SubDict = dict[str, RunDict]
Doubles = tuple[float, float] | list[float, float] | ndarray[(2,), float]
Signal = base.BaseRaw | BaseEpochs | Evoked
ListNum = int | float | ndarray | list | tuple


@njit(cache=True)
def rand_seed(seed: int = 42):
    random.seed(seed)
