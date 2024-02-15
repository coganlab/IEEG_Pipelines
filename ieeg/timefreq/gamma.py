from functools import singledispatch

import numpy as np
from mne import Epochs
from mne.io import Raw, base
from tqdm import tqdm
from numpy.typing import NDArray
from scipy.fft import fft, ifft
from tqdm.auto import trange
from joblib import Parallel, delayed

from ieeg.process import COLA, cpu_count, get_mem, parallelize
from ieeg.timefreq.utils import BaseEpochs, Evoked, Signal
from hilbert import filterbank_hilbert_first_half_wrapper, extract_channel_wrapper

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

    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> from bids import BIDSLayout
    >>> from ieeg.navigate import trial_ieeg
    >>> from ieeg.io import raw_from_layout
    >>> from ieeg.timefreq.utils import crop_pad
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=50)
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1",
    ... preload=True, extension=".vhdr", verbose=False) # doctest: +ELLIPSIS
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> trials = trial_ieeg(raw, "AD1-4, ATT1,2", (-0.5, 1),
    ... preload=True, verbose=False, picks=['AD2'])
    >>> gamma = extract(trials, n_jobs=1)
    >>> crop_pad(gamma, "0.5s") # doctest: +ELLIPSIS
    <Epochs |  1 events (all good), 0 – 0.5 s, baseline off, ~44 kB,...
     'AD1-4, ATT1,2': 1>
    >>> gamma.resample(100, verbose=50) # doctest: +ELLIPSIS
    <Epochs |  1 events (all good), 0 – 0.49 s, baseline off, ~41 kB,...
     'AD1-4, ATT1,2': 1>
    >>> expected = np.array([
    ... 3.5729, 3.8323, 4.0820, 5.4100, 8.0623, 12.579, 20.280, 31.027, 43.918,
    ... 56.523, 65.739, 68.678, 64.378, 54.357, 42.245, 32.177, 26.568, 23.578,
    ... 20.584, 17.003, 13.105, 9.6693, 6.9391, 4.8671, 3.6392, 3.0246, 2.8268,
    ... 2.9109, 3.2376, 3.6906, 4.1659, 4.5842, 4.9949, 5.3240, 6.0321, 7.1968,
    ... 8.0531, 8.4710, 8.3094, 7.8219, 7.3717, 7.1496, 7.0281, 7.0632, 7.0525,
    ... 7.2997, 7.7566, 7.7874, 7.3208, 6.4729]) * 1e-05
    >>> np.abs(np.sum(gamma._data - expected)) < 1e-6
    True
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
        if n_jobs != 1:
            ins = (in_data[trial].T for trial in trials)
            par_out = parallelize(filterbank_hilbert, ins, fs=fs, Wn=passband,
                                  n_jobs=n_jobs)
            env[:, :, :] = np.array([np.sum(out, axis=-1).T for
                                     _, out, _ in par_out])
        else:
            if verbose:
                trials = tqdm(trials)
            for trial in trials:
                _, out, _ = filterbank_hilbert(in_data[trial, :, :].T, fs,
                                               passband, 1)
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
    """Extract gamma band envelope from Raw object."""
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs,
                         verbose=verbose)


@extract.register
def _(inst: BaseEpochs, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1, verbose: bool = True) -> Epochs:
    """Extract gamma band envelope from Epochs object."""
    return _extract_inst(inst, fs, copy, passband=passband, n_jobs=n_jobs,
                         verbose=verbose)


@extract.register
def _(inst: Evoked, fs: int = None,
      passband: tuple[int, int] = (70, 150),
      copy: bool = True, n_jobs=-1, verbose: bool = True) -> Evoked:
    """Extract gamma band envelope from Evoked object."""
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


def filterbank_hilbert(x, fs, Wn=[70, 150], n_jobs=1):
    '''
    Compute the phase and amplitude (envelope) of a signal for a single
    frequency band, as in [#edwards]_. This is done using a filter bank of
    gaussian shaped filters with center frequencies linearly spaced until 4Hz
    and then logarithmically spaced. The Hilbert Transform of each filter's
    output is computed and the amplitude and phase are computed from the
    complex values. See [#edwards]_ for details on the filter bank used.

    See Also
    --------
    filter_hilbert

    Parameters
    ----------
    x : np.ndarray, shape (time, channels)
        Signal to filter. Filtering is performed on each channel independently.
    fs : int
        Sampling rate.
    Wn : list or array-like, length 2, default=[70, 150]
        Lower and upper boundaries for filterbank center frequencies. A range
        of [1, 150] results in 42 filters.
    n_jobs : int, default=1
        Number of jobs to use to compute filterbank across channels in
        parallel.

    Returns
    -------
    x_phase : np.ndarray, shape (time, channels, frequency_bins)
        Phase of each frequency bin in the filter bank for each channel.
    x_envelope : np.ndarray, shape (time, channels, frequency_bins)
        Envelope of each frequency bin in the filter bank for each channel.
    center_freqs : np.ndarray, shape (frequency_bins,)
        Center frequencies for each frequency bin used in the filter bank.
    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(1000,3) # 3 channels of signals
    >>> fs = 500
    >>> x_phase, x_envelope, freqs = filterbank_hilbert(x, fs, Wn=[1, 150])
    >>> # the outputs have the phase and envelope for each channel and each
    filter in the filterbank
    >>> x_phase.shape  # 3rd dimension is one for each filter in filterbank
    (1000, 3, 42)
    >>> x_envelope.shape
    (1000, 3, 42)
    >>> round(freqs[0], 8) # center frequency of first filter bank filter
    1.21558792
    >>> round(freqs[-1], 8) # center frequency of last filter bank filter
    143.97075186

    '''

    x = x.astype('float32')
    minf, maxf = Wn
    
    if minf >= maxf:
        raise ValueError(
            (f'Upper bound of frequency range must be greater than lower bound'
             f', but got lower bound of {minf} and upper bound of {maxf}'))

    if x.ndim != 1 and x.ndim != 2:
        raise ValueError('Input signal must be 1- or 2-dimensional but got input with'
             f'shape {x.shape}')
    
    Xf, freqs, cfs, N, sds, h, x = filterbank_hilbert_first_half_wrapper(x, fs, minf, maxf)

    def extract_channel(Xf):
        return extract_channel_wrapper(Xf, freqs, cfs, N, sds, h, x, Wn)
    
    # pre-allocate
    hilb_phase = np.zeros((*x.shape, len(cfs)), dtype='float32')
    hilb_amp = np.zeros((*x.shape, len(cfs)), dtype='float32')

    # process channels sequentially
    if n_jobs == 1:
        for chn in range(x.shape[1]):
            hilb_phase[:, chn], hilb_amp[:, chn] = extract_channel(Xf[:, chn])
    # process channels in parallel
    else:
        results = Parallel(n_jobs)(delayed(extract_channel)(
            Xf[:, chn]) for chn in range(x.shape[1]))
        for chn, (phase, amp) in enumerate(results):
            hilb_phase[:, chn], hilb_amp[:, chn] = phase, amp

    return hilb_phase, hilb_amp, cfs

