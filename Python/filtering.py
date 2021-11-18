import mne
import np
from typing import Union


def line_filter(data: Union[np.ndarray, list]):
    raw = mne.io.read_raw(data)
    filt = mne.chpi.filter_chpi(raw)
    return filt