from collections import Counter
from typing import Union, List
import argparse

import numpy as np
from mne.io import pick
from mne.utils import logger, _pl, verbose
from numpy.typing import ArrayLike
from scipy import stats
from tqdm import tqdm

import sys
import os
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from PreProcess.timefreq import multitaper, fastmath, \
    utils as mt_utils  # noqa: E402
from PreProcess.utils.utils import is_number  # noqa: E402

ListNum = Union[int, float, np.ndarray, list, tuple]


@verbose
def line_filter(raw: mt_utils.Signal, fs: float = None, freqs: ListNum = 60.,
                filter_length: str = '10s', notch_widths: ListNum = 10.,
                mt_bandwidth: float = None, p_value: float = 0.05,
                picks: list[Union[int, str]] = None, n_jobs: int = None,
                adaptive: bool = True, low_bias: bool = True,
                copy: bool = True, *, verbose: Union[int, bool, str] = None
                ) -> mt_utils.Signal:
    """Line noise notch filter for the signal instance.

    Applies a multitaper power line noise notch filter to the signal, operating
    on the last dimension. Uses the F-test to find significant sinusoidal
    components to remove. This is done by fitting a sinusoid to the power
    spectrum at each time point and frequency, and testing whether the
    resulting fit is significantly different from a flat spectrum. The
    significance test is done using an F-test, which requires fitting two
    models (one flat, one sinusoidal) at each time point and frequency. The
    F-test is corrected for multiple comparisons using the Benjamini-Hochberg
    procedure.

    Parameters
    ----------
    raw : mt_utils.Signal
        Signal to filter.
    fs : float, optional
        Sampling rate in Hz. Default is taken from the raw object.
    freqs : float | array-like of float, optional
        Frequencies to notch filter in Hz, e.g. np.arange(60, 241, 60).
        None can only be used with the mode 'spectrum_fit', where an F
        test is used to find sinusoidal components.
    filter_length : str | int, optional
        Length of the filter to use. If str, assumed to be human-readable time
        in units of "s" or "ms" (e.g., "10s" or "5500ms"). If an int, it
        is assumed to be in samples and used directly.
    notch_widths : float | array of float, optional
        Width of the stop band (centred at each freq in freqs) in Hz.
        Default is 10.
    mt_bandwidth : float, optional
        The bandwidth of the multitaper windowing function in Hz. Default will
        set the half frequency bathwidth to 4 Hz.
    p_value : float, optional
        P-value to use in F-test thresholding to determine significant
        sinusoidal components to remove. Note that this will be Bonferroni
        corrected for the number of frequencies, so large p-values may be
        justified.
    picks : list of int | list of str, optional
        Channels to filter. If None, all channels will be filtered.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is number of cores on
        machine.
    adaptive : bool, optional
        Use adaptive weights to combine the tapered spectra into PSD.
        Default is True.
    low_bias : bool, optional
        Only use tapers with more than 90 percent spectral concentration within
        bandwidth. Default is True.
    copy : bool, optional
        If True, a copy of x, filtered, is returned. Otherwise, it operates
        on x in place.
    verbose : bool, str, int, or None, optional
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    filt : mt_utils.Signal
        The signal instance with the filtered data.

    See Also
    --------
    <https://mne.tools/stable/generated/mne.filter.notch_filter.html>

    Notes
    -----
    The frequency response is (approximately) given by
    ::
        1-|----------         -----------
          |          \\       /
          |           \\     /
          |            \\   /
          |             \\ /
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
    cite this in publications if this function is used.
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

    data_idx = np.where([ch_t in set(raw.get_channel_types(
        only_data_chs=True)) for ch_t in raw.get_channel_types()])[0]

    # convert filter length to samples
    if filter_length is None:
        filter_length = x.shape[-1]

    filter_length: int = min(mt_utils.to_samples(filter_length, fs),
                             x.shape[-1])

    # Define adaptive windowing function
    def get_window_thresh(n_times: int = filter_length
                          ) -> tuple(np.ndarray, float):
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


def mt_spectrum_proc(x: np.ndarray, sfreq: float, line_freqs: ListNum,
                     notch_widths: ListNum, picks: list, n_jobs: int,
                     get_wt: callable) -> np.ndarray:
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


@verbose
def _mt_remove_win(x: np.ndarray, sfreq: float, line_freqs: ListNum,
                   notch_width: ListNum, get_thresh: callable,
                   n_jobs: int = None, verbose: bool = None
                   ) -> tuple(np.ndarray, List[float]):
    """Remove line frequencies from data using multitaper method."""
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

    mt_utils.COLA(process, store, n_times, n_samples, n_overlap, sfreq,
                  n_jobs=n_jobs, verbose=verbose).feed(x)
    assert idx[0] == n_times
    return x_out, rm_freqs


def _mt_remove(x: np.ndarray, sfreq: float, line_freqs: ListNum,
               notch_widths: ListNum, window_fun: np.ndarray,
               threshold: float, get_thresh: callable,
               ) -> tuple(np.ndarray, List[float]):
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
        ranges = [(freq - notch_width / 2, freq + notch_width / 2
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


def _prep_for_filtering(x: np.ndarray, picks: list = None
                        ) -> tuple(np.ndarray, tuple, int):
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
            """,
        epilog="""
            Made by Aaron Earle-Richardson (ae166@duke.edu)
            """)

    parser.add_argument("-s", "--subject", required=False, default=None,
                        help="data subject to clean")
    return parser


def main(subject: str = None):
    import mne
    from bids import BIDSLayout
    from PreProcess.navigate import raw_from_layout, LAB_root, save_derivative

    # %% Set up logging
    mne.set_log_file("output.log",
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")

    bids_root = LAB_root + "/BIDS-1.0_SentenceRep/BIDS"
    layout = BIDSLayout(bids_root)
    if subject is not None:
        do_subj = [subject]
    else:
        do_subj = layout.get(return_type="id", target="subject")
    do_subj.sort()
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        taskIDs = [int(os.environ['SLURM_ARRAY_TASK_ID'])]
    else:
        taskIDs = list(range(len(do_subj)))
    for id in taskIDs:
        subj = do_subj[id]
        try:
            raw = raw_from_layout(layout, subject=subj, extension=".edf",
                                  preload=False)
            # %% filter data
            filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1,
                               filter_length='700ms', verbose=10,
                               freqs=[60], notch_widths=20)
            filt2 = line_filter(filt, mt_bandwidth=10., n_jobs=-1,
                                filter_length='20s', verbose=10,
                                freqs=[60, 120, 180, 240], notch_widths=20)
            # %% Save the data
            save_derivative(filt2, layout, "clean")
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
