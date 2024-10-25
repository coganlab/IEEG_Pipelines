from os import PathLike as PL

import mne.io
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base
from numpy import ndarray

PathLike = str | PL
RunDict = dict[int, mne.io.Raw]
SubDict = dict[str, RunDict]
Doubles = tuple[float, float] | list[float, float] | ndarray[(2,), float]
Signal = base.BaseRaw | BaseEpochs | Evoked
ListNum = int | float | ndarray | list | tuple
