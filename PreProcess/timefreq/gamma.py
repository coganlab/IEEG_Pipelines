from naplib.preprocessing import filterbank_hilbert
from PreProcess.timefreq.utils import BaseEpochs, Evoked, Signal, COLA
from PreProcess.utils.utils import get_mem, cpu_count
from mne.io import base, Raw
from mne import Epochs
from functools import singledispatch
import numpy as np


@singledispatch
def extract(data: np.ndarray, fs: int, passband: tuple[int, int] = (70, 150),
            copy: bool = True, n_jobs=-1) -> np.ndarray:
    """Extract gamma band envelope from data.

    Parameters
    ----------
    data : (np.ndarray, shape (..., channels)) | Signal
        Data to extract gamma envelope from. If Signal, will use the _data
        attribute.
    fs : int, optional
        Sampling frequency of data. If Signal, will use the data.info['sfreq'].
        Otherwise, must be provided.
    passband : tuple[int, int], optional
        Passband in Hz, high gamma band by default (70, 150)
    copy : bool, optional
        Whether to copy data or operate in place if False, by default True
    n_jobs : int, optional
        Number of jobs to run in parallel, by default all available cores

    Returns
    -------
    np.ndarray
        Gamma envelope.
    """

    if copy:
        in_data = data.copy()
    else:
        in_data = data

    passband = list(passband)
    env = np.zeros(in_data.shape)

    if len(in_data.shape) == 3:  # Assume shape is (trials, channels, time)
        for trial in range(in_data.shape[0]):
            _, env[trial, :, :], _ = filterbank_hilbert(in_data[trial, :, :].T,
                                                        fs, passband, n_jobs)
    elif len(in_data.shape) == 2:  # Assume shape is (channels, time)
        _, env, _ = filterbank_hilbert(in_data.T, fs, passband, n_jobs)
    else:
        raise ValueError("number of dims should be either 2 or 3, not {}"
                         "".format(len(in_data.shape)))

    return env.T


def _extract_inst(inst: Signal, fs: int, copy: bool, **kwargs) -> Signal:
    if fs is None:
        fs = inst.info['sfreq']
    if copy:
        sig = inst.copy()
    else:
        sig = inst

    sig._data = extract(sig._data, fs, copy=False, **kwargs)

    return sig


@extract.register
def _(inst: base.BaseRaw, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1) -> Raw:
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs)


@extract.register
def _(inst: BaseEpochs, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1) -> Epochs:
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs)


@extract.register
def _(inst: Evoked, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1) -> Evoked:
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs)

