"""
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

PathLike = str | PL
RunDict = dict[int, mne.io.Raw]
SubDict = dict[str, RunDict]
Doubles = tuple[float, float] | list[float, float] | ndarray[(2,), float]
Signal = base.BaseRaw | BaseEpochs | Evoked | _BaseTFR
ListNum = int | float | ndarray | list | tuple

# import sys
# import os
# from pathlib import Path  # if you haven't already done so
#
# file = Path(__file__).resolve()
# parent, root = file.parent, file.parents[1]
# sys.path.append(str(root))
#
# # Additionally remove the current file's directory from sys.path
# try:
#     sys.path.remove(str(parent))
# except ValueError:  # Already removed
#     pass
