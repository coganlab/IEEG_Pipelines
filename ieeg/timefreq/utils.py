from typing import Union

from mne.epochs import BaseEpochs
from mne.io import base
from mne.time_frequency import EpochsTFRArray
from mne.utils import fill_doc, verbose
from mne import Evoked

from ieeg import Signal
from ieeg.process import ensure_int, validate_type
from ieeg.calc.oversample import resample
from joblib import delayed, Parallel
import numpy as np
# from ieeg.process import parallelize
# from scipy.fft import fft, ifft, fftfreq
# from functools import partial
# from mne.evoked import Evoked
# from mne.time_frequency import tfr_array_stockwell as tas


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


# @verbose
# def cwt(inst: BaseEpochs, f_low: float, f_high: float, n_fft: int,
#         width: float = 1.0, n_jobs: int = 1, decim: int = 1, verbose=10
#         ) -> EpochsTFRArray:
#     """Compute the wavelet scaleogram.
#
#
#
#     Parameters
#     ----------
#     inst : instance of Raw, Epochs, or Evoked
#         The instance to compute the wavelet scaleogram for.
#     f_low : float
#         The lowest frequency to compute the scaleogram for.
#     f_high : float
#         The highest frequency to compute the scaleogram for.
#     k0 : int
#         The wavelet parameter.
#     n_jobs : int
#         The number of jobs to run in parallel.
#     decim : int
#         The decimation factor.
#     verbose : int
#         The verbosity level.
#
#     Returns
#     -------
#     scaleogram : instance of EpochsTFR
#         The wavelet scaleogram.
#
#     Notes
#     -----
#     Similar to https://www.mathworks.com/help/wavelet/ref/cwt.html
#
#     Examples
#     --------
#     >>> import mne
#     >>> from ieeg.io import raw_from_layout
#     >>> from ieeg.navigate import trial_ieeg
#     >>> from bids import BIDSLayout
#     # >>> with mne.use_log_level(0):
#     >>> bids_root = mne.datasets.epilepsy_ecog.data_path()
#     >>> layout = BIDSLayout(bids_root)
#     >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
#     ... extension=".vhdr", verbose=False)
#     >>> epochs = trial_ieeg(raw, ['AST1,3', 'G16'], (-1, 2), verbose=False)
#     >>> cwt(epochs, n_jobs=1, decim=10, n_fft=40) # doctest: +ELLIPSIS
#     Using data from preloaded Raw for 2 events and 3001 original time...
#         Getting epoch for 85800-88801
#         Getting epoch for 90760-93761
#     0 bad epochs dropped
#     Data is self data: False
#     <TFR from Epochs, unknown method | 2 epochs × 98 channels × 46 freqs ...
#
#     """
#
#     ins = ((ti, chi, a[None, None]) for ti, b in enumerate(inst) for chi,
#     a in enumerate(b))
#     proc = Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem",
#                     return_as="generator_unordered")(
#         delayed(tas_wrap)(i, j, x, inst.info['sfreq'], f_low, f_high, n_fft,
#                           width, decim, False, 1) for i, j, x in ins)
#     out0, _, freqs = tas(inst[0, 0], inst.info['sfreq'], f_low, f_high,
#     n_fft,
#                          width, decim, False, 1)
#
#     out = np.empty((len(inst), len(inst.ch_names), *out0.data.shape),
#     dtype=np.float32)
#     out[0, 0] = out0
#     for ti, chi, (x, _, _) in proc:
#         out[ti, chi] = x
#
#     return EpochsTFRArray(inst.info, out, inst.times[::decim], freqs)
#     # data = inst.get_data(copy=False)
#     # wave, _, freqs = tfr_array_stockwell(data, inst.info['sfreq'], f_low,
#     f_high,
#     #                              n_jobs=n_jobs, decim=decim,
#     return_itc=False,
#     #                              average=False, width=width,
#     verbose=verbose,
#     #                                      n_fft=n_fft)
#     #
#     # return EpochsTFRArray(inst.info, wave, inst.times[::decim], freqs)
#
#
# def tas_wrap(i, j, *args, **kwargs):
#     return i, j, tas(*args, **kwargs)

@verbose
def wavelet_scaleogram(inst: BaseEpochs, f_low: float = 2,
                       f_high: float = 1000, k0: int = 6, n_jobs: int = 1,
                       decim: int = 1, verbose=10) -> EpochsTFRArray:
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
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> epochs = trial_ieeg(raw, ['AST1,3', 'G16'], (-1, 2), verbose=False)
    >>> wavelet_scaleogram(epochs, n_jobs=1, decim=10) # doctest: +ELLIPSIS
    Using data from preloaded Raw for 2 events and 3001 original time points...
        Getting epoch for 85800-88801
        Getting epoch for 90760-93761
    0 bad epochs dropped
    Data is self data: False
    <TFR from Epochs, unknown method | 2 epochs × 98 channels × 46 freqs × ...

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

    proc = Parallel(n_jobs=n_jobs, verbose=verbose, require="sharedmem",
                    return_as="generator_unordered")(delayed(_ifft_abs)(x, i)
                                                     for x, i in ins)
    for _ in proc:
        pass
    # parallelize(_ifft_abs, ins, require='sharedmem', n_jobs=n_jobs,
    #             verbose=verbose)

    return EpochsTFRArray(inst.info, wave, inst.times[::decim], 1 / period,
                          events=inst.events, event_id=inst.event_id)


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


# def _check_input_st(x_in, n_fft):
#     """Aux function."""
#     # flatten to 2 D and memorize original shape
#     n_times = x_in.shape[-1]
#
#     def _is_power_of_two(n):
#         return not (n > 0 and (n & (n - 1)))
#
#     if n_fft is None or (not _is_power_of_two(n_fft) and n_times > n_fft):
#         # Compute next power of 2
#         n_fft = 2 ** int(np.ceil(np.log2(n_times)))
#     elif n_fft < n_times:
#         raise ValueError(
#             f"n_fft cannot be smaller than signal size. Got {n_fft} <
#             {n_times}."
#         )
#     if n_times < n_fft:
#         # logger.info(
#         #     f'The input signal is shorter ({x_in.shape[-1]}) than "n_fft"
#         ({n_fft}). '
#         #     "Applying zero padding."
#         # )
#         zero_pad = n_fft - n_times
#         pad_array = np.zeros(x_in.shape[:-1] + (zero_pad,), x_in.dtype)
#         x_in = np.concatenate((x_in, pad_array), axis=-1)
#     else:
#         zero_pad = 0
#     return x_in, n_fft, zero_pad
#
#
# def _precompute_st_windows(n_samp, start_f, stop_f, sfreq, width):
#     """Precompute stockwell Gaussian windows (in the freq domain)."""
#     tw = fftfreq(n_samp, 1.0 / sfreq) / n_samp
#     tw = np.r_[tw[:1], tw[1:][::-1]]
#
#     k = width  # 1 for classical stowckwell transform
#     f_range = np.arange(start_f, stop_f, 1)
#     windows = np.empty((len(f_range), len(tw)), dtype=np.complex128)
#     for i_f, f in enumerate(f_range):
#         if f == 0.0:
#             window = np.ones(len(tw))
#         else:
#             window = (f / (np.sqrt(2.0 * np.pi) * k)) * np.exp(
#                 -0.5 * (1.0 / k**2.0) * (f**2.0) * tw**2.0
#             )
#         window /= window.sum()  # normalisation
#         windows[i_f] = fft(window)
#     return windows
#
#
# def _st_power_itc(x, start_f, compute_itc, zero_pad, decim, W, average):
#     """Aux function."""
#     decim = slice(None, None, decim)
#     n_samp = x.shape[-1]
#     decim_indices = decim.indices(n_samp - zero_pad)
#     n_out = len(range(*decim_indices))
#     out_shape = (len(W), n_out) if average else (x.shape[0], len(W), n_out)
#     psd = np.empty(out_shape)
#     itc = np.empty_like(psd) if compute_itc else None
#     X = fft(x)
#     XX = np.concatenate([X, X], axis=-1)
#     for i_f, window in enumerate(W):
#         f = start_f + i_f
#         ST = ifft(XX[:, f : f + n_samp] * window)
#         TFR = ST[:, slice(*decim_indices)]
#         TFR_abs = np.abs(TFR)
#         TFR_abs[TFR_abs == 0] = 1.0
#         if compute_itc:
#             TFR /= TFR_abs
#             itc[i_f] = np.abs(np.mean(TFR, axis=0))
#         TFR_abs *= TFR_abs
#         if average:
#             psd[i_f] = np.mean(TFR_abs, axis=0)
#         else:
#             psd[..., i_f, :] = TFR_abs
#     return psd, itc
#
#
# def _compute_freqs_st(fmin, fmax, n_fft, sfreq):
#     """Compute the frequencies for the Stockwell transform.
#
#     Parameters
#     ----------
#     fmin
#     fmax
#     n_fft
#     sfreq
#
#     Returns
#     -------
#     start_f
#     stop_f
#     freqs
#
#     Examples
#     --------
#     >>> _compute_freqs_st(30, 500, 200, 2048)
#
#     """
#     from scipy.fft import fftfreq
#
#     freqs = fftfreq(n_fft, 1.0 / sfreq)
#     if fmin is None:
#         fmin = freqs[freqs > 0][0]
#     if fmax is None:
#         fmax = freqs.max()
#
#     start_f = np.abs(freqs - fmin).argmin()
#     stop_f = np.abs(freqs - fmax).argmin()
#     freqs = freqs[start_f:stop_f]
#     return start_f, stop_f, freqs
#
#
# def compute_freqs_st(fmin, fmax, n_fft, sfreq):
#     """Compute the frequencies for the Stockwell transform.
#
#     Parameters
#     ----------
#     fmin
#     fmax
#     n_fft
#     sfreq
#
#     Returns
#     -------
#     start_f
#     stop_f
#     freqs
#
#     Examples
#     --------
#     >>> compute_freqs_st(30, 500, 40, 2048)
#
#     """
#     start_f, stop_f, freqs = _compute_freqs_st(fmin, fmax, n_fft, sfreq)
#     temp = n_fft
#     while stop_f - start_f < n_fft:
#         temp += 1
#         start_f, stop_f, freqs = _compute_freqs_st(fmin, fmax, temp, sfreq)
#     return start_f, stop_f, freqs
#
#
# @verbose
# def tfr_array_stockwell(
#     data,
#     sfreq,
#     fmin=None,
#     fmax=None,
#     n_fft=None,
#     width=1.0,
#     decim=1,
#     return_itc=False,
#     n_jobs=None,
#     average=True,
#     *,
#     verbose=None,
# ):
#     """Compute power and intertrial coherence using Stockwell (S) transform.
#
#     Same computation as `~mne.time_frequency.tfr_stockwell`, but operates on
#     :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.
#
#     See :footcite:`Stockwell2007,MoukademEtAl2014,WheatEtAl2010,JonesEtAl2006
#     for more information.
#
#     Parameters
#     ----------
#     data : ndarray, shape (n_epochs, n_channels, n_times)
#         The signal to transform.
#     sfreq : float
#         The sampling frequency.
#     fmin : None, float
#         The minimum frequency to include. If None defaults to the minimum fft
#         frequency greater than zero.
#     fmax : None, float
#         The maximum frequency to include. If None defaults to the maximum fft
#     n_fft : int | None
#         The length of the windows used for FFT. If None, it defaults to the
#         next power of 2 larger than the signal length.
#     width : float
#         The width of the Gaussian window. If < 1, increased temporal
#         resolution, if > 1, increased frequency resolution. Defaults to 1.
#         (classical S-Transform).
#     %(decim_tfr)s
#     return_itc : bool
#         Return intertrial coherence (ITC) as well as averaged power.
#     %(n_jobs)s
#     %(verbose)s
#
#     Returns
#     -------
#     st_power : ndarray
#         The multitaper power of the Stockwell transformed data.
#         The last two dimensions are frequency and time.
#     itc : ndarray
#         The intertrial coherence. Only returned if return_itc is True.
#     freqs : ndarray
#         The frequencies.
#
#     See Also
#     --------
#     mne.time_frequency.tfr_stockwell
#     mne.time_frequency.tfr_multitaper
#     mne.time_frequency.tfr_array_multitaper
#     mne.time_frequency.tfr_morlet
#     mne.time_frequency.tfr_array_morlet
#
#     References
#     ----------
#     .. footbibliography::
#     """
#     if data.ndim != 3:
#         raise ValueError(
#             "data must be 3D with shape (n_epochs, n_channels, n_times), "
#             f"got {data.shape}"
#         )
#
#     trials, n_channels, n_out = data[..., ::decim].shape
#     start_f, stop_f, freqs = compute_freqs_st(fmin, fmax, n_fft, sfreq)
#
#     W = _precompute_st_windows(data.shape[-1], start_f, stop_f, sfreq, width)
#     n_freq = stop_f - start_f
#     out_shape = (n_channels, n_freq, n_out) if average else \
#         (trials, n_channels, n_freq, n_out)
#     psd = np.empty(out_shape)
#     itc = np.empty(out_shape) if return_itc else None
#
#     ins = (data[:, c, :] for c in range(n_channels))
#     myfunc = partial(_st_power_itc, start_f=start_f, compute_itc=return_itc,
#                      zero_pad=0, decim=decim, W=W, average=average)
#     tfrs = Parallel(n_jobs=n_jobs, verbose=verbose, return_as='generator')(
#         delayed(myfunc)(x) for x in ins)
#     for c, (this_psd, this_itc) in enumerate(iter(tfrs)):
#         psd[..., c, :, :] = this_psd
#         if this_itc is not None:
#             itc[c] = this_itc
#
#     return psd, itc, freqs
#
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


def resample_tfr(tfr, sfreq, o_sfreq=None, copy=False):
    """Resample a TFR object to a new sampling frequency"""
    if copy:
        tfr = tfr.copy()

    if o_sfreq is None:
        # o_sfreq = len(tfr.times) / (tfr.tmax - tfr.tmin)
        o_sfreq = tfr.info["sfreq"]

    tfr._data = resample(tfr._data, o_sfreq, sfreq, axis=-1)
    lowpass = tfr.info.get("lowpass")
    lowpass = np.inf if lowpass is None else lowpass
    with tfr.info._unlock():
        tfr.info["lowpass"] = min(lowpass, sfreq / 2)
        tfr.info["sfreq"] = sfreq
    new_times = resample(tfr.times, o_sfreq, sfreq, axis=-1)
    # adjust indirectly affected variables
    tfr._set_times(new_times)
    tfr._raw_times = tfr.times
    tfr._update_first_last()
    return tfr


if __name__ == "__main__":
    # Description: Produce spectrograms for each subject
    from ieeg.io import get_data, raw_from_layout
    from ieeg.calc.scaling import rescale
    from ieeg.viz.ensemble import chan_grid
    from ieeg.viz.parula import parula_map
    from ieeg.navigate import trial_ieeg, crop_empty_data, \
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

    data = dict()
    for sub in subjects:
        if sub != "D0005":
            continue
        # Load the data
        filt = raw_from_layout(layout.derivatives['notch'], subject=sub,
                               extension='.edf', desc='notch', preload=False)

        # Crop raw data to minimize processing time
        good = crop_empty_data(filt, ).copy()

        # good.info['bads'] = channel_outlier_marker(good, 3, 2)
        good.drop_channels(good.info['bads'])
        good.load_data()

        ch_type = filt.get_channel_types(only_data_chs=True)[0]
        good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

        # Remove intermediates from mem
        # good.plot()

        # epoching and trial outlier removal

        save_dir = os.path.join(layout.root, 'derivatives', 'spec',
                                'wavelet_test', sub)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch, t, name in zip(
                ("Start", "Word/Response/LS"),
                ((-0.5, 0), (-1, 1)),
                (
                        "base", "resp")):
            trials = trial_ieeg(good, epoch, t, preload=False)
            n_samps = trials.times.shape[-1]
            n_fft = 2 ** int(np.ceil(np.log2(n_samps)))
            times = [None, None]
            offset = n_fft - n_samps / 2 / good.info['sfreq']
            times[0] = t[0] - offset
            times[1] = t[1] + offset
            trials = trial_ieeg(good, epoch, times, preload=True)
            outliers_to_nan(trials, outliers=10)
            spec = wavelet_scaleogram(trials, n_jobs=-2, decim=int(
                good.info['sfreq'] / 100))
            # spec = cwt(trials, 30, 500, n_fft, 1, -1, 4)
            crop_pad(spec, "0.5s")
            # spec = spec.decimate(2, 1)
            del trials

            # if spec.sfreq % 100 == 0:
            #     factor = spec.sfreq // 100
            #     offset = len(spec.times) % factor
            #     spec = spec.decimate(factor, offset)
            resample_tfr(spec, 100,
                         spec.times.shape[0] / (spec.tmax - spec.tmin))

            if name == "base":
                base = spec.copy().crop(-0.5, 0)
            else:
                data[name] = spec

    # Plot the spectrogram
    result = rescale(data['resp'], base, mode='ratio', copy=True)
    avg = result.average(lambda x: np.nanmean(x, axis=0))
    chan_grid(avg, size=(20, 10), vlim=(-.5, 3.), cmap=parula_map)
