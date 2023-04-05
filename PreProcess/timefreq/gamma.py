from naplib.preprocessing import filterbank_hilbert
from PreProcess.timefreq.utils import BaseEpochs, Evoked, Signal, COLA
from PreProcess.utils.utils import get_mem, cpu_count
from mne.io import base, Raw
from mne import Epochs
from functools import singledispatch
from tqdm import tqdm
import numpy as np


@singledispatch
def extract(data: np.ndarray, fs: int = None,
            passband: tuple[int, int] = (70, 150), copy: bool = True,
            n_jobs=-1, verbose: bool = True) -> np.ndarray:
    """Extract gamma band envelope from data.

    Parameters
    ----------
    data : (np.ndarray, shape ((epochs) ,channels, samples)) | Signal
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

    Notes
    -----
    This function is a wrapper for
    `filterbank_hilbert <https://naplib-python.readthedocs.io/en/latest/referen
    ces/preprocessing.html#naplib.preprocessing.filterbank_hilbert>`_. It is a
    convenience function for extracting the gamma envelope from data. It is
    optimized for speed, but not memory. If you have a lot of data, you may
    want to epoch your data first and then extract the envelope.
    """

    if fs is None:
        raise ValueError("fs must be provided if data is not a Signal")

    if copy:
        in_data = data.copy()
    else:
        in_data = data

    passband = list(passband)
    env = np.zeros(in_data.shape)

    if len(in_data.shape) == 3:  # Assume shape is (trials, channels, time)
        trials = range(in_data.shape[0])
        if verbose:
            trials = tqdm(trials)
        for trial in trials:
            _, out, _ = filterbank_hilbert(in_data[trial, :, :].T, fs,
                                           passband, n_jobs)
            env[trial, :, :] = np.sum(out, axis=-1).T
    elif len(in_data.shape) == 2:  # Assume shape is (channels, time)
        _, out, _ = filterbank_hilbert(in_data.T, fs, passband, n_jobs)
        env = np.sum(out, axis=-1).T
    else:
        raise ValueError("number of dims should be either 2 or 3, not {}"
                         "".format(len(in_data.shape)))

    return env


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
      copy: bool = True, n_jobs=-1, verbose: bool = True) -> Raw:
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs,
                         verbose=verbose)


@extract.register
def _(inst: BaseEpochs, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1, verbose: bool = True) -> Epochs:
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs,
                         verbose=verbose)


@extract.register
def _(inst: Evoked, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1, verbose: bool = True) -> Evoked:
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs,
                         verbose=verbose)


def _my_hilt(x: np.ndarray, fs, Wn=(1, 150), n_jobs=-1):

    # Set default window function and threshold
    cfs = get_centers(Wn)
    n_times = x.shape[0]
    bytes_per = 16  # numpy complex128 is 16 bytes per num
    chunk_size = min([x.size * bytes_per * len(cfs), get_mem()])
    n_samples = int(chunk_size / (cpu_count() * x.shape[1] * len(cfs)))
    n_overlap = (n_samples + 1) // 2
    x_out = np.zeros_like(x.T)
    idx = [0]

    # Define how to process a chunk of data
    def process(x_):
        out = filterbank_hilbert(x_, fs, Wn, 1)
        return (out[1])  # must return a tuple

    # Define how to store a chunk of fully processed data (it's trivial)
    def store(x_):
        stop = idx[0] + x_.T.shape[-1]
        x_out[..., idx[0]:stop] += x_.T
        idx[0] = stop

    COLA(process, store, n_times, n_samples, n_overlap, fs,
         n_jobs=n_jobs).feed(x)
    assert idx[0] == n_times
    return x_out, cfs


def get_centers(Wn):
    """Get center frequencies for filter bank.

    Parameters
    ----------
    Wn : tuple
        Frequency range to use for filter bank.

    Returns
    -------
    cfs : list
        Center frequencies for filter bank.
        """

    # create filter bank
    a = np.array([np.log10(0.39), 0.5])
    f0 = 0.018
    octSpace = 1. / 7
    minf, maxf = Wn
    if minf >= maxf:
        raise ValueError(
            f'Upper bound of frequency range must be greater than lower '
            f'bound, but got lower bound of {minf} and upper bound of {maxf}')
    maxfo = np.log2(maxf / f0)  # octave of max freq

    cfs = [f0]
    sigma_f = 10 ** (a[0] + a[1] * np.log10(cfs[-1]))

    while np.log2(cfs[-1] / f0) < maxfo:

        if cfs[-1] < 4:
            cfs.append(cfs[-1] + sigma_f)
        else:  # switches to log spacing at 4 Hz
            cfo = np.log2(cfs[-1] / f0)  # current freq octave
            cfo += octSpace  # new freq octave
            cfs.append(f0 * (2 ** (cfo)))

        sigma_f = 10 ** (a[0] + a[1] * np.log10(cfs[-1]))

    cfs = np.array(cfs)
    if np.logical_and(cfs >= minf, cfs <= maxf).sum() == 0:
        raise ValueError(
            f'Frequency band is too narrow, so no filters in filterbank are '
            f'placed inside. Try a wider frequency band.')

    cfs = cfs[np.logical_and(cfs >= minf,
                             cfs <= maxf)]  # choose those that lie in the
    # input freqRange
    return cfs
