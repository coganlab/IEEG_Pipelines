from collections import Counter
from typing import TypeVar, Union, List

import numpy as np
from mne.io import pick
from mne.utils import logger, _pl, verbose
from numpy.typing import ArrayLike
from scipy import stats
from tqdm import tqdm

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from PreProcess.timefreq import multitaper, fastmath, utils as mt_utils
from PreProcess.utils.utils import is_number


ListNum = TypeVar("ListNum", int, float, np.ndarray, list, tuple)


@verbose
def line_filter(raw: mt_utils.Signal, fs: float = None, freqs: ListNum = None,
                filter_length: str = 'auto',
                notch_widths: Union[ListNum, int] = None,
                mt_bandwidth: float = None, p_value: float = 0.05,
                picks: ListNum = None, n_jobs: int = None,
                adaptive: bool = True, low_bias: bool = True,
                copy: bool = True, *, verbose: Union[int, bool, str] = None
                ) -> mt_utils.Signal:
    r"""Notch filter for the signal x.
    Applies a multitaper notch filter to the signal x, operating on the last
    dimension.
    Parameters
    ----------
    raw : array
        Signal to filter.
    fs : float
        Sampling rate in Hz.
    freqs : float | array of float | None
        Frequencies to notch filter in Hz, e.g. np.arange(60, 241, 60).
        None can only be used with the mode 'spectrum_fit', where an F
        test is used to find sinusoidal components.
    %(filter_length_notch)s
    notch_widths : float | array of float | None
        Width of the stop band (centred at each freq in freqs) in Hz.
        If None, freqs / 200 is used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'spectrum_fit' mode.
    p_value : float
        P-value to use in F-test thresholding to determine significant
        sinusoidal components to remove when method='spectrum_fit' and
        freqs=None. Note that this will be Bonferroni corrected for the
        number of frequencies, so large p-values may be justified.
    %(picks_nostr)s
        Only supported for 2D (n_channels, n_times) and 3D
        (n_epochs, n_channels, n_times) data.
    %(n_jobs_fir)s
    copy : bool
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    %(phase)s
    %(verbose)s
    Returns
    -------
    xf : array
        The x array filtered.
    See Also
    --------
    filter_data
    resample
    Notes
    -----
    The frequency response is (approximately) given by::
        1-|----------         -----------
          |          \       /
      |H| |           \     /
          |            \   /
          |             \ /
        0-|              -
          |         |    |    |         |
          0        Fp1 freq  Fp2       Nyq
    For each freq in freqs, where ``Fp1 = freq - trans_bandwidth / 2`` and
    ``Fs2 = freq + trans_bandwidth / 2``.
    References
    ----------
    Multi-taper removal is inspired by code from the Chronux toolbox, see
    www.chronux.org and the book "Observed Brain Dynamics" by Partha Mitra
    & Hemant Bokil, Oxford University Press, New York, 2008. Please
    cite this in publications if method 'spectrum_fit' is used.
    """
    if fs is None:
        fs = raw.info["sfreq"]
    if copy:
        filt = raw.copy()
    else:
        filt = raw
    x = mt_utils._check_filterable(filt.get_data("data"), 'notch filtered',
                          'notch_filter')
    if freqs is not None:
        freqs = np.atleast_1d(freqs)
        # Only have to deal with notch_widths for non-autodetect
        if notch_widths is None:
            notch_widths = freqs / 200.0
        elif np.any(notch_widths < 0):
            raise ValueError('notch_widths must be >= 0')
        else:
            notch_widths = np.atleast_1d(notch_widths)
            if len(notch_widths) == 1:
                notch_widths = notch_widths[0] * np.ones_like(freqs)
            elif len(notch_widths) != len(freqs):
                raise ValueError('notch_widths must be None, scalar, or the '
                                 'same length as freqs')

    data_idx = [ch_t in set(raw.get_channel_types(only_data_chs=True)
                            ) for ch_t in raw.get_channel_types()]

    # convert filter length to samples
    if isinstance(filter_length, str) and filter_length == 'auto':
        filter_length = '10s'
    if filter_length is None:
        filter_length = x.shape[-1]

    filter_length: int = min(mt_utils.to_samples(filter_length, fs),
                             x.shape[-1])

    # Define adaptive windowing function
    def get_window_thresh(n_times: int = filter_length) -> (ArrayLike, float):
        # figure out what tapers to use
        window_fun, _, _ = multitaper.params(n_times, fs, mt_bandwidth,
                                             low_bias, adaptive,
                                             verbose=verbose)

        # F-stat of 1-p point
        threshold = stats.f.ppf(1 - p_value / n_times, 2,
                                2 * len(window_fun) - 2)
        return window_fun, threshold

    filt._data[data_idx] = mt_spectrum_proc(
        x, fs, freqs, notch_widths, picks, n_jobs, get_window_thresh)

    return filt


def mt_spectrum_proc(x: ArrayLike, sfreq: float, line_freqs: ListNum,
                     notch_widths: ListNum, picks: list, n_jobs: int,
                     get_wt: callable) -> ArrayLike:
    """Call _mt_spectrum_remove."""
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, picks)

    # Execute channel wise sine wave detection and filtering
    freq_list = list()
    for ii, x_ in tqdm(enumerate(x), position=0, total=x.shape[0],
                       leave=True, desc='channels'):
        logger.debug(f'Processing channel {ii}')
        if ii in picks:
            x[ii], f = _mt_remove_win(x_, sfreq, line_freqs, notch_widths,
                                      get_wt, n_jobs)
            freq_list.append(f)

    # report found frequencies, but do some sanitizing first by binning into
    # 1 Hz bins
    counts = Counter(sum((np.unique(np.round(ff)).tolist()
                          for f in freq_list for ff in f), list()))
    kind = 'Detected' if line_freqs is None else 'Removed'
    found_freqs = '\n'.join(f'    {freq:6.2f} : '
                            f'{counts[freq]:4d} window{_pl(counts[freq])}'
                            for freq in sorted(counts)) or '    None'
    logger.info(f'{kind} notch frequencies (Hz):\n{found_freqs}')

    x.shape = orig_shape
    return x


def _mt_remove_win(x: np.ndarray, sfreq: float, line_freqs: ListNum,
                   notch_width: ListNum, get_thresh: callable,
                   n_jobs: int = None) -> (ArrayLike, List[float]):
    # Set default window function and threshold
    window_fun, thresh = get_thresh()
    n_times = x.shape[-1]
    n_samples = window_fun.shape[1]
    n_overlap = (n_samples + 1) // 2
    x_out = np.zeros_like(x)
    rm_freqs = list()
    idx = [0]

    # Define how to process a chunk of data
    def process(x_):
        out = _mt_remove(x_, sfreq, line_freqs, notch_width, window_fun,
                         thresh, get_thresh)
        rm_freqs.append(out[1])
        return (out[0],)  # must return a tuple

    # Define how to store a chunk of fully processed data (it's trivial)
    def store(x_):
        stop = idx[0] + x_.shape[-1]
        x_out[..., idx[0]:stop] += x_
        idx[0] = stop

    mt_utils._COLA(process, store, n_times, n_samples, n_overlap, sfreq,
                   n_jobs=n_jobs, verbose=True).feed(x)
    assert idx[0] == n_times
    return x_out, rm_freqs


def _mt_remove(x: np.ndarray, sfreq: float, line_freqs: ListNum,
               notch_widths: ListNum, window_fun: np.ndarray,
               threshold: float, get_thresh: callable,
               ) -> (ArrayLike, List[float]):
    """Use MT-spectrum to remove line frequencies.
    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """

    assert x.ndim == 1
    if x.shape[-1] != window_fun.shape[-1]:
        window_fun, threshold = get_thresh(x.shape[-1])
    # compute mt_spectrum (returning n_ch, n_tapers, n_freq)
    x_p, freqs = multitaper.spectra(x[np.newaxis, :], window_fun, sfreq)
    f_stat, A = fastmath.sine_f_test(window_fun, x_p)

    # find frequencies to remove
    indices = np.where(f_stat > threshold)[1]
    # pdf = 1-stats.f.cdf(f_stat, 2, window_fun.shape[0]-2)
    # indices = np.where(pdf < 1/x.shape[-1])[1]
    # specify frequencies within indicated ranges
    if line_freqs is not None and notch_widths is not None:
        if not isinstance(notch_widths, (list, tuple)) and is_number(
                notch_widths):
            notch_widths = [notch_widths] * len(line_freqs)
        ranges = [(freq - notch_width/2, freq + notch_width/2
                   ) for freq, notch_width in zip(line_freqs, notch_widths)]
        indices = [ind for ind in indices if any(
            lower <= freqs[ind] <= upper for (lower, upper) in ranges)]

    fits = list()
    # make "time" vector
    rads = 2 * np.pi * (np.arange(x.size) / float(sfreq))
    for ind in indices:
        c = 2 * A[0, ind]
        fit = np.abs(c) * np.cos(freqs[ind] * rads + np.angle(c))
        fits.append(fit)

    if len(fits) == 0:
        datafit = 0.0
    else:
        # fitted sinusoids are summed, and subtracted from data
        datafit = np.sum(fits, axis=0)

    return x - datafit, freqs[indices]


def _prep_for_filtering(x: ArrayLike, picks: list = None) -> ArrayLike:
    """Set up array as 2D for filtering ease."""
    x = mt_utils._check_filterable(x)
    orig_shape = x.shape
    x = np.atleast_2d(x)
    picks = pick._picks_to_idx(x.shape[-2], picks)
    x.shape = (np.prod(x.shape[:-1]), x.shape[-1])
    if len(orig_shape) == 3:
        n_epochs, n_channels, n_times = orig_shape
        offset = np.repeat(np.arange(0, n_channels * n_epochs, n_channels),
                           len(picks))
        picks = np.tile(picks, n_epochs) + offset
    elif len(orig_shape) > 3:
        raise ValueError('picks argument is not supported for data with more'
                         ' than three dimensions')
    assert all(0 <= pick < x.shape[0] for pick in picks)  # guaranteed by above

    return x, orig_shape, picks


if __name__ == "__main__":
    import mne
    from bids import BIDSLayout
    from PreProcess.navigate import raw_from_layout, get_data, open_dat_file
    import PreProcess.utils.plotting


    # %% Set up logging
    mne.set_log_file("output.log",
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")

    bids_root = mne.datasets.epilepsy_ecog.data_path()
    layout = BIDSLayout(bids_root)
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    raw.load_data()
    layout, raw, D_dat_raw, D_dat_filt = get_data(29, "SentenceRep")
    filt = mne.io.read_raw_fif(layout.root + "/derivatives/sub-D00" + str(
        29) + "_" + "SentenceRep" + "_filt_ieeg.fif")

    # %% filter data
    # filt = line_filter(raw, mt_bandwidth=10.0, n_jobs=-1,
    #                    filter_length='700ms', verbose=10,
    #                    freqs=[60], notch_widths=20, p_value=.05)
    # filt2 = line_filter(filt, mt_bandwidth=10.0, n_jobs=-1,
    #                     filter_length='20s', verbose=10,
    #                     freqs=[120, 180, 240], notch_widths=20, p_value=.05)

    # # %% plot results
    # data = [raw, filt, filt2, raw_dat, dat]
    # figure_compare(data, ["BIDS Un", "BIDS 700ms ", "BIDS 20s+700ms ", "Un",
    #                       ""], avg=True, verbose=10, proj=True, fmax=250)
    # figure_compare(data, ["BIDS Un", "BIDS 700ms ", "BIDS 20s+700ms ", "Un",
    #                       ""], avg=False, verbose=10, proj=True, fmax=250)

    # %% plot results
    # params = dict(method='multitaper', fmin=55, fmax=65, tmax=200,
    #               bandwidth=0.5, n_jobs=8)
    # fpsd = filt.compute_psd(**params)
    # fpsd.plot()

    # %% plot results
    is_sent = False
    annot = None
    for event in filt.annotations:
        if event['description'] in ['Audio']:
            if event['duration'] > 1:
                is_sent = True
            else:
                is_sent = False
        if event['description'] not in ['Listen', ':=:']:
            if is_sent:
                trial_type = "Sentence/"
            else:
                trial_type = "Word/"
        else:
            trial_type = "Start/"
        event['description'] = trial_type + event['description']
        if annot is None:
            annot = mne.Annotations(**event)
        else:
            event.pop('orig_time')
            annot.append(**event)
    filt.set_annotations(annot)
    freqs = np.arange(10, 200., 2.)
    spect = multitaper.spectrogram(filt, 'Word/Audio', -0.5, 1.5, 'Start', -0.5, 0, freqs, n_jobs=-1, verbose=10)
    spect.plot()