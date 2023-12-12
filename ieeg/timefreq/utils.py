from typing import Union

import numpy as np
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.io import base
from mne.time_frequency import EpochsTFR, tfr
from mne.utils import fill_doc, verbose

from ieeg import Signal
from ieeg.process import ensure_int, parallelize, validate_type


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


@verbose
def wavelet_scaleogram(inst: BaseEpochs, f_low: float = 2,
                       f_high: float = 1000, k0: int = 6, n_jobs: int = 1,
                       decim: int = 1, verbose=10) -> EpochsTFR:
    """Compute the wavelet scaleogram.



    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The instance to compute the wavelet scaleogram for.
    f_low : float
        The lowest frequency to compute the scaleogram for.
    f_high : float
        The highest frequency to compute the scaleogram for.
    k0 : int
        The wavelet parameter.
    n_jobs : int
        The number of jobs to run in parallel.
    decim : int
        The decimation factor.
    verbose : int
        The verbosity level.

    Returns
    -------
    scaleogram : instance of EpochsTFR
        The wavelet scaleogram.

    Notes
    -----
    Similar to https://www.mathworks.com/help/wavelet/ref/cwt.html

    Examples
    --------
    >>> import mne
    >>> from ieeg.io import raw_from_layout
    >>> from ieeg.navigate import trial_ieeg
    >>> from bids import BIDSLayout
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path()
    >>> seeg = mne.io.read_raw(mne.datasets.misc.data_path() / 'seeg' /
    ... 'sample_seeg_ieeg.fif')  # doctest: +ELLIPSIS
    Opening raw data file ...
        Range : 1310640 ... 1370605 =   1311.411 ...  1371.411 secs
    Ready.
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> epochs = trial_ieeg(raw, ['AST1,3', 'G16'], (-1, 2), verbose=True
    ... ) # doctest: +ELLIPSIS
    Used Annotations descriptions: ['AD1-4, ATT1,2', 'AST1,3', 'G16', 'PD', ...
    Not setting metadata
    2 matching events found
    No baseline correction applied
    0 projection items activated
    >>> wavelet_scaleogram(epochs, n_jobs=-2, decim=10) # doctest: +ELLIPSIS
    Using data from preloaded Raw for 2 events and 3001 original time points...
        Getting epoch for 85800-88801
        Getting epoch for 90760-93761
    0 bad epochs dropped
    Data is self data: False
    Not setting metadata
    <EpochsTFR | time : [-1.000000, 2.000000], freq : [2.079716, 1064.814640...

    """
    data = inst.get_data(copy=False)

    f = np.fft.fft(data - np.mean(data, axis=-1, keepdims=True))

    daughter, period = calculate_wavelets(inst.info['sfreq'], f_high, f_low,
                                          data.shape[-1], k0)

    wave = np.empty((f.shape[0], f.shape[1], len(period),
                     data[..., ::decim].shape[-1]), dtype=np.float64)
    # ch X trials X freq X time
    ins = ((f[:, None, i], i) for i in range(f.shape[1]))

    def _ifft_abs(x, i):
        np.abs(np.fft.ifft(x * np.tile(daughter, (
            f.shape[0], 1, 1)))[..., ::decim], out=wave[:, i])

    parallelize(_ifft_abs, ins, require='sharedmem', n_jobs=n_jobs,
                verbose=verbose)

    return EpochsTFR(inst.info, wave, inst.times[::decim], 1 / period)


def calculate_wavelets(sfreq: float, f_high: float, f_low: float,
                       n_samples: int, k0: int = 6):
    """Calculate Morlet wavelets for a range of frequencies.

    Parameters
    ----------
    sfreq : float
        The sampling frequency.
    f_high : float
        The highest frequency to compute the scaleogram for.
    f_low : float
        The lowest frequency to compute the scaleogram for.
    n_samples : int
        The number of samples.
    k0 : int
        The wavelet parameter. Indicates the center frequency of the wavelet.

    Returns
    -------
    daughter : ndarray
        The wavelets.
    period : ndarray
        The periods.

    Examples
    --------
    >>> daughter, period = calculate_wavelets(1000, 100, 2, 1000)
    >>> daughter.shape
    (29, 1001)
    >>> period.shape
    (29,)
    """

    dt = 1 / sfreq
    s0 = 1 / (f_high + (0.1 * f_high))  # the smallest resolvable scale
    n = n_samples
    J1 = (np.log2(n * dt / s0)) / 0.2  # (J1 determines the largest scale)

    k = np.arange(np.fix(n / 2)) + 1
    k = k * ((2 * np.pi) / (n * dt))
    kr = (-k).tolist()
    kr.reverse()
    k = np.array([0] + k.tolist() + kr)

    scale = s0 * np.power(2., (np.arange(0, J1) * 0.2))
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.square(k0)))
    period = fourier_factor * scale
    xxx = np.min(np.where((1. / period) < f_low))
    period = np.flip(period[:xxx])
    scale = np.flip(scale[:xxx])

    scale1 = scale
    period = fourier_factor * scale1

    expnt = -np.square(scale1[:, None] * k[None, :] - k0) / 2. * (k > 0.)
    norm = np.sqrt(scale1 * k[2]) * (np.power(np.pi, (-0.25))) * np.sqrt(n)
    daughter = norm[:, None] * np.exp(expnt)
    daughter = daughter * (k > 0.)

    return daughter, period


def roundup(x: float) -> int:
    """Round up to the nearest integer."""
    n, d = divmod(x, 1)
    return int(n) + (d > 0)


@verbose
def cwt(inst: BaseEpochs, f_low: float = 2,
        f_high: float = 1000, n_jobs: int = 1, k0: int = 6,
        decim: int = 1, verbose=10) -> EpochsTFR:
    """Compute the wavelet scaleogram.



    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The instance to compute the wavelet scaleogram for.
    f_low : float
        The lowest frequency to compute the scaleogram for.
    f_high : float
        The highest frequency to compute the scaleogram for.
    k0 : int
        The wavelet parameter.
    n_jobs : int
        The number of jobs to run in parallel.
    decim : int
        The decimation factor.
    verbose : int
        The verbosity level.

    Returns
    -------
    scaleogram : instance of EpochsTFR
        The wavelet scaleogram.

    Notes
    -----
    Similar to https://www.mathworks.com/help/wavelet/ref/cwt.html
    """
    data = inst.get_data()  # (trials X channels X timepoints)
    data -= np.mean(data, axis=-1, keepdims=True)

    # Ws, period = calculate_wavelets(inst.info['sfreq'], f_high, f_low,
    # data.shape[-1])
    freqs = np.logspace(np.log10(f_low), np.log10(f_high), 50)
    common_factor = (data.shape[-1] + 1) * np.pi / 5 / inst.info['sfreq']
    n_cycles = np.min(freqs) * common_factor
    wavelets = tfr.morlet(inst.info['sfreq'], freqs, n_cycles=n_cycles,
                          zero_mean=True)
    wave = np.empty((data.shape[0], data.shape[1], len(freqs),
                     roundup(data.shape[2] / decim)), dtype=np.float64)

    def _cwt(x, i):
        wave[i] = tfr.cwt(x, wavelets, use_fft=True, decim=decim)

    # perform the cwt across trials and channels
    # iterate through fisrt and second dimension
    ins = ((d, i) for i, d in enumerate(data))
    parallelize(_cwt, ins, require='sharedmem', n_jobs=n_jobs,
                verbose=verbose)

    return EpochsTFR(inst.info, wave, inst.times[::decim], freqs)


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


if __name__ == "__main__":
    # Description: Produce spectrograms for each subject
    from ieeg.io import get_data, raw_from_layout
    from ieeg.navigate import trial_ieeg, channel_outlier_marker, \
        crop_empty_data, \
        outliers_to_nan
    import os
    import numpy as np

    # check if currently running a slurm job
    HOME = os.path.expanduser("~")

    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        LAB_root = os.path.join(HOME, "workspace", "CoganLab")
        layout = get_data("SentenceRep", root=LAB_root)
        subjects = list(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:  # if not then set box directory
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
        layout = get_data("SentenceRep", root=LAB_root)
        subjects = layout.get(return_type="id", target="subject")

    for sub in subjects:
        if sub != "D0029":
            continue
        # Load the data
        filt = raw_from_layout(layout.derivatives['clean'], subject=sub,
                               extension='.edf', desc='clean', preload=False)

        # Crop raw data to minimize processing time
        good = crop_empty_data(filt, ).copy()

        # good.info['bads'] = channel_outlier_marker(good, 3, 2)
        good.drop_channels(good.info['bads'])
        good.load_data()

        ch_type = filt.get_channel_types(only_data_chs=True)[0]
        good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

        # Remove intermediates from mem
        good.plot()

        # epoching and trial outlier removal

        save_dir = os.path.join(layout.root, 'derivatives', 'spec',
                                'wavelet_test', sub)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch, t, name in zip(
                ("Start", "Word/Response/LS", "Word/Audio/LS", "Word/Audio/LM",
                 "Word/Audio/JL", "Word/Go/LS", "Word/Go/LM",
                 "Word/Go/JL"),
                ((-0.5, 0), (-1, 1), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5),
                 (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)),
                (
                        "base", "resp", "aud_ls", "aud_lm", "aud_jl", "go_ls",
                        "go_lm",
                        "go_jl")):
            times = [None, None]
            times[0] = t[0] - 0.5
            times[1] = t[1] + 0.5
            trials = trial_ieeg(good, epoch, times, preload=True)
            outliers_to_nan(trials, outliers=10)
            # spec = wavelet_scaleogram(trials, n_jobs=-2, decim=int(
            #     good.info['sfreq'] / 100))
            spec = cwt(trials, n_jobs=-2, decim=int(
                good.info['sfreq'] / 100))
