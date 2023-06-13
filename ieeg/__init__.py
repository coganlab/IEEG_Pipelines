""" ieeg: a package for working with intracranial EEG data
.. include:: ../ieeg/README.md
    :parser: myst_parser.sphinx_
"""
import mne.io
from os import PathLike as PL
from numpy import ndarray
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base
from mne.time_frequency import _BaseTFR

# from . import (  # noqa: F401
#     calc,
#     timefreq,
#     viz,
#     io,
#     mt_filter,
#     navigate,
#     process)

PathLike = str | PL
RunDict = dict[int, mne.io.Raw]
SubDict = dict[str, RunDict]
Doubles = tuple[float, float] | list[float, float] | ndarray[(2,), float]
Signal = base.BaseRaw | BaseEpochs | Evoked | _BaseTFR
ListNum = int | float | ndarray | list | tuple
