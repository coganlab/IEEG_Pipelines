from typing import Union

import numpy as np
from mne.utils import fill_doc
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base

from ieeg.utils.process import validate_type, ensure_int
from ieeg import Signal


def to_samples(time_length: Union[str, int], sfreq: float) -> int:
    """Convert a time length to a number of samples.

    Parameters
    ----------
    time_length : str | int
        The time length to convert. If a string, it must be a human-readable
        time, e.g. "10s".
    sfreq : float
        The sampling frequency.

    Returns
    -------
    samples : int
        The number of samples.
    """
    validate_type(time_length, (str, int))
    if isinstance(time_length, str):
        time_length = time_length.lower()
        err_msg = ('filter_length, if a string, must be a '
                   'human-readable time, e.g. "0.7s", or "700ms", not '
                   '"%s"' % time_length)
        low = time_length.lower()
        if low.endswith('us'):
            mult_fact = 1e-6
            time_length = time_length[:-2]
        elif low.endswith('ms'):
            mult_fact = 1e-3
            time_length = time_length[:-2]
        elif low[-1] == 's':
            mult_fact = 1
            time_length = time_length[:-1]
        elif low.endswith('sec'):
            mult_fact = 1
            time_length = time_length[:-3]
        elif low[-1] == 'm':
            mult_fact = 60
            time_length = time_length[:-1]
        elif low.endswith('min'):
            mult_fact = 60
            time_length = time_length[:-3]
        else:
            raise ValueError(err_msg)
        # now get the number
        try:
            time_length = float(time_length)
        except ValueError:
            raise ValueError(err_msg)
        time_length = max(int(np.ceil(time_length * mult_fact * sfreq)), 1)
    time_length = ensure_int(time_length, 'filter_length')
    return time_length


@fill_doc
def crop_pad(inst: Signal, pad: str, copy: bool = False) -> Signal:
    """Crop and pad an instance.

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The instance to crop and pad.
    pad : str
        The amount of time to pad the instance. If a string, it must be a
        human-readable time, e.g. "10s".
    copy : bool, optional
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place. Defaults to False.

    Returns
    -------
    inst : instance of Raw, Epochs, or Evoked
        The cropped and de-padded instance.
    """
    if copy:
        out = inst.copy()
    else:
        out = inst
    pad = to_samples(pad, inst.info['sfreq']) / inst.info['sfreq']
    out.crop(tmin=inst.tmin + pad, tmax=inst.tmax - pad)
    return out


def _check_filterable(x: Union[Signal, np.ndarray],
                      kind: str = 'filtered',
                      alternative: str = 'filter') -> np.ndarray:
    # Let's be fairly strict about this -- users can easily coerce to ndarray
    # at their end, and we already should do it internally any time we are
    # using these low-level functions. At the same time, let's
    # help people who might accidentally use low-level functions that they
    # shouldn't use by pushing them in the right direction
    if isinstance(x, (base.BaseRaw, BaseEpochs, Evoked)):
        try:
            name = x.__class__.__name__
        except Exception:
            pass
        else:
            raise TypeError(
                'This low-level function only operates on np.ndarray '
                f'instances. To get a {kind} {name} instance, use a method '
                f'like `inst_new = inst.copy().{alternative}(...)` '
                'instead.')
    validate_type(x, (np.ndarray, list, tuple))
    x = np.asanyarray(x)
    if x.dtype != np.float64:
        raise ValueError('Data to be %s must be real floating, got %s'
                         % (kind, x.dtype,))
    return x
