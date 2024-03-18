from os import PathLike as PL

import mne.io
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base
from numba import njit
from numpy import ndarray, random

PathLike = str | PL
RunDict = dict[int, mne.io.Raw]
SubDict = dict[str, RunDict]
Doubles = tuple[float, float] | list[float, float] | ndarray[(2,), float]
Signal = base.BaseRaw | BaseEpochs | Evoked
ListNum = int | float | ndarray | list | tuple


@njit(cache=True)
def _rand_seed(seed: int = 42):
    random.seed(seed)
