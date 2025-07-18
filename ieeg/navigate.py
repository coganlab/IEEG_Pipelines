import mne
import numpy as np
from bids import BIDSLayout
from mne.utils import fill_doc, verbose
from scipy.signal import detrend
from sklearn.neighbors import LocalOutlierFactor

from ieeg import Doubles, Signal
from ieeg.calc import stats
from ieeg.io import update
from ieeg.timefreq.utils import to_samples


def crop_empty_data(raw: mne.io.Raw, bound: str = 'boundary',
                    start_pad: str = "10s", end_pad: str = "10s"
                    ) -> mne.io.Raw:
    """Crops out long stretches of data with no events.

    Takes raw instance with annotated events and crops the instance so that the
    raw file starts at start_pad before the first event and stops an amount of
    time in seconds given by end_pad after the last event.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw file to crop.
    bound : str, optional
        The annotation description to use as a boundary, by default 'boundary'
    start_pad : str, optional
        The amount of time to pad the start of the file, by default "10s"
    end_pad : str, optional
        The amount of time to pad the end of the file, by default "10s"

    Returns
    -------
    mne.io.Raw
        The cropped raw file.

    Examples
    --------
    >>> import mne
    >>> from ieeg.io import raw_from_layout
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=False)
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> cropped = crop_empty_data(raw, 'onset')
    >>> cropped.times[0], cropped.times[-1]
    (0.0, 104.94)
    """

    crop_list = []

    start_pad = to_samples(start_pad, raw.info['sfreq']) / raw.info['sfreq']
    end_pad = to_samples(end_pad, raw.info['sfreq']) / raw.info['sfreq']

    # split annotations into blocks
    annot = raw.annotations.copy()
    block_idx = [idx + 1 for idx, val in
                 enumerate(annot) if bound in val['description']]
    block_annot = [annot[i: j] for i, j in
                   zip([0] + block_idx, block_idx +
                       ([len(annot)] if block_idx[-1] != len(annot) else []))]

    for block_an in block_annot:
        # remove boundary events from annotations
        no_bound = None
        for an in block_an:
            if bound not in an['description']:
                if no_bound is None:
                    no_bound = mne.Annotations(**an)
                else:
                    an.pop('orig_time')
                    no_bound.append(**an)

        # Skip if block is all boundary events
        if no_bound is None:
            continue
        # get start and stop time from raw.annotations onset attribute
        t_min = max(0, no_bound.onset[0] - start_pad)
        t_max = no_bound.onset[-1] + end_pad

        # create new cropped raw file
        crop_list.append(raw.copy().crop(tmin=t_min, tmax=t_max))

    return mne.concatenate_raws(crop_list)


@fill_doc
@verbose
def channel_outlier_marker(input_raw: Signal, outlier_sd: float = 3,
                           max_rounds: int = np.inf, axis: int = 0,
                           save: bool = False, verbose: bool = True
                           ) -> list[str]:
    """Identify bad channels by variance.

    Parameters
    ----------
    input_raw : Signal
        Raw data to be analyzed.
    outlier_sd : int, optional
        Number of standard deviations above the mean to be considered an
        outlier, by default 3
    max_rounds : int, optional
        Maximum number of variance estimations, by default runs until no
        more bad channels are found.
    axis : int, optional
        Axis to calculate variance over, by default 0
    save : bool, optional
        Whether to save bad channels to raw.info['bads'], by default False
    %(verbose)s

    Returns
    -------
    list[str]
        List of bad channel names.

    Examples
    --------
    >>> import mne
    >>> from ieeg.io import raw_from_layout
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=False)
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> bads = channel_outlier_marker(raw, 3, 2)
    outlier round 1 channels: ['AST2']
    outlier round 1 channels: ['AST2', 'RQ2']
    outlier round 1 channels: ['AST2', 'RQ2', 'N/A']
    outlier round 2 channels: ['AST2', 'RQ2', 'N/A', 'G32']
    outlier round 2 channels: ['AST2', 'RQ2', 'N/A', 'G32', 'AD3']
    outlier round 2 channels: ['AST2', 'RQ2', 'N/A', 'G32', 'AD3', 'PD4']
    """

    names = input_raw.copy().pick('data').ch_names
    data = detrend(input_raw.get_data('data'))  # channels X time
    bads = []  # output for bad channel names
    desc = []  # output for bad channel descriptions

    # Pop out names to bads output using comprehension list
    for ind, i in stats.outlier_repeat(data, outlier_sd, max_rounds, axis):
        bads.append(names[ind])
        desc.append(f'outlier round {i} more than {outlier_sd} SDs above mean')
        # log channels excluded per round
        if verbose:
            mne.utils.logger.info(f'outlier round {i} channels: {bads}')

    if save:
        if not hasattr(input_raw, 'filenames'):
            raise ValueError("Raw instance must have filenames attribute to "
                             "save bad channels")
        for file in input_raw.filenames:
            update(file, bads, desc)

    return bads


def find_bad_channels_lof(
    raw,
    *,
    picks=None,
    metric="seuclidean",
    threshold=1.5,
    return_scores=False,
    **kwargs
):
    """Find bad channels using Local Outlier Factor (LOF) algorithm.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to process.
    n_neighbors : int
        Number of neighbors defining the local neighborhood (default is 20).
        Smaller values will lead to higher LOF scores.
    %(picks_good_data)s
    metric : str
        Metric to use for distance computation. Default is “euclidean”,
        see :func:`sklearn.metrics.pairwise.distance_metrics` for details.
    threshold : float
        Threshold to define outliers. Theoretical threshold ranges anywhere
        between 1.0 and any positive integer. Default: 1.5
        It is recommended to consider this as an hyperparameter to optimize.
    return_scores : bool
        If ``True``, return a dictionary with LOF scores for each
        evaluated channel. Default is ``False``.
    %(verbose)s

    Returns
    -------
    noisy_chs : list
        List of bad M/EEG channels that were automatically detected.
    scores : ndarray, shape (n_picks,)
        Only returned when ``return_scores`` is ``True``. It contains the
        LOF outlier score for each channel in ``picks``.

    See Also
    --------
    maxwell_filter
    annotate_amplitude

    Notes
    -----
    See :footcite:`KumaravelEtAl2022` and :footcite:`BreunigEtAl2000` for background on
    choosing ``threshold``.

    .. versionadded:: 1.7

    References
    ----------
    .. footbibliography::
    """  # noqa: E501

    if metric == "seuclidean":
        kwargs.setdefault("metric_params",
                          {"V": np.var(raw.get_data(), axis=0)})
    # Get the channel types
    picks = list(range(len(raw.ch_names))) if picks is None else picks
    ch_names = [raw.ch_names[pick] for pick in picks]
    data = raw.get_data(picks=picks)
    clf = LocalOutlierFactor(n_neighbors=len(raw.ch_names) // 5, metric=metric,
                             **kwargs)
    bad_channel_indices = picks
    clf.fit_predict(data)
    scores_lof = clf.negative_outlier_factor_
    while len(bad_channel_indices) / len(picks) > 0.2:
        bad_channel_indices = [
            i for i, v in enumerate(np.abs(scores_lof)) if v >= threshold
        ]
        threshold += 1
    bads = [ch_names[idx] for idx in bad_channel_indices]
    if return_scores:
        return bads, scores_lof
    else:
        return bads


@verbose
def outliers_to_nan(trials: mne.epochs.BaseEpochs, outliers: float,
                    copy: bool = False, picks: list = 'data',
                    deviation: callable = np.nanstd,
                    center: callable = np.nanmean, tmin: int | float = None,
                    tmax: int | float = None, verbose=None
                    ) -> mne.epochs.BaseEpochs:
    """Set outliers to nan.

    Parameters
    ----------
    trials : mne.epochs.BaseEpochs
        The trials to remove outliers from.
    outliers : float
        The number of deviations above the mean to be considered an outlier.
    copy : bool, optional
        Whether to copy the data, by default False
    picks : list, optional
        The channels to remove outliers from, by default 'data'
    deviation: callable, optional
        Metric function to determine the deviation from the center. Default is
        median absolute deviation.
    center : callable, optional
        Metric function to determine the center of the data. Default is median.

    Returns
    -------
    mne.epochs.BaseEpochs
        The trials with outliers set to nan.

    Examples
    --------
    >>> import mne
    >>> from ieeg.io import raw_from_layout
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=50)
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> epochs = trial_ieeg(raw, ['AD1-4, ATT1,2', 'AST1,3', 'G16', 'PD'],
    ... (-1, 2), preload=True, verbose=False)
    >>> outliers_to_nan(epochs, 1, True, [0], verbose=False,
    ... ).get_data()[1] # doctest: +ELLIPSIS
    array([[            nan,             nan,             nan, ...,
                        nan,             nan,             nan],
           [-4.63276969e-04, -4.67964469e-04, -4.72261344e-04, ...,
             1.41019078e-04,  1.22269102e-04,  9.92222578e-05],
           [-2.84374563e-04, -3.03515188e-04, -3.08593313e-04, ...,
             9.57034922e-05,  5.19535000e-05,  1.40628818e-05],
           ...,
           [-4.69516375e-04, -5.09750688e-04, -5.69906813e-04, ...,
             3.45716687e-04,  3.10951125e-04,  3.25794844e-04],
           [-1.67187703e-04, -1.95703313e-04, -2.23047047e-04, ...,
            -2.52734531e-04, -2.89062656e-04, -2.57422031e-04],
           [-1.98796781e-04, -2.79265281e-04, -3.31218250e-04, ...,
            -2.73129219e-05, -1.52703172e-04, -2.52702875e-04]])
    >>> outliers_to_nan(epochs, .1, verbose=False, copy=True,
    ... deviation=None).get_data()[0] # doctest: +SKIP
    """
    if copy:
        trials = trials.copy()
    picks = mne.io.pick._picks_to_idx(trials.info, picks)
    if isinstance(trials, mne.time_frequency.BaseTFR):
        data = trials.get_data(picks, tmin=tmin, tmax=tmax)
        out_data = trials.get_data(picks)
    else:
        if not isinstance(trials, mne.epochs.EpochsArray):
            trials.load_data()
        data = trials.get_data(picks, tmin=tmin, tmax=tmax, verbose=verbose,
                               copy=False)
        out_data = trials.get_data(picks, verbose=False, copy=False)

    # bool array of where to keep data trials X channels
    # if deviation is None or center is None:
    #     keep = stats.find_outliers_lof(data, outliers)
    # else:
    keep = stats.find_outliers(data, outliers, deviation, center)

    # set outliers to nan if not keep
    data = np.where(keep[..., None], out_data, np.nan)
    trials._data[:, picks] = data

    return trials


@fill_doc
@verbose
def trial_ieeg(raw: mne.io.Raw, event: str | list[str, ...], times: Doubles,
               verbose=None, **kwargs) -> mne.Epochs:
    """Epochs data from a mne Raw iEEG instance.

    Takes a mne Raw instance and epochs the data around a specified event. If
    baseline is specified, the data is also epoched around the baseline event
    and the baseline is subtracted from the data epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to epoch.
    event : str
        The event to epoch around.
    times : tuple[float, float]
        The time window to epoch around the event.
    %(picks_all)s
    %(reject_epochs)s
    %(flat)s
    %(decim)s
    %(epochs_reject_tmin_tmax)s
    %(detrend_epochs)s
    %(proj_epochs)s
    %(on_missing_epochs)s
    %(verbose)s

    Returns
    -------
    mne.Epochs
        The epoched data.

    Examples
    --------
    >>> import mne
    >>> from ieeg.io import raw_from_layout
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=False)
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    >>> epochs = trial_ieeg(raw, "AD1-4, ATT1,2", (-1, 2), verbose=True
    ... ) # doctest: +ELLIPSIS
    Used Annotations descriptions: ['AD1-4, ATT1,2', 'AST1,3', 'G16', 'PD',...
    Not setting metadata
    1 matching events found
    No baseline correction applied
    0 projection items activated
    >>> epochs = trial_ieeg(raw, ['AST1,3', 'G16'], (-1, 2), verbose=True
    ... ) # doctest: +ELLIPSIS
    Used Annotations descriptions: ['AD1-4, ATT1,2', 'AST1,3', 'G16', 'PD', ...
    Not setting metadata
    2 matching events found
    No baseline correction applied
    0 projection items activated
    """

    # determine the events
    events, ids = mne.events_from_annotations(raw)
    dat_ids = [ids[i] for i in mne.event.match_event_names(ids, event)]
    rev = {k: v for k, v in ids.items() if v in dat_ids}

    # epoch the data
    return mne.Epochs(raw, events, event_id=rev, tmin=times[0],
                      tmax=times[1], baseline=None, verbose=verbose, **kwargs)


if __name__ == "__main__":
    from os import path
    from ieeg.io import raw_from_layout

    # %% Set up logging
    log_filename = "output.log"
    # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    HOME = path.expanduser("~")
    LAB_root = path.join(HOME, "Box", "CoganLab")
    TASK = "SentenceRep"
    sub_num = 29
    subj = "D" + str(sub_num).zfill(4)
    # layout, raw, D_dat_raw, D_dat_filt = get_data(sub_num, TASK)
    bids_root = LAB_root + "/BIDS-1.4_SentenceRep/BIDS"
    layout = BIDSLayout(bids_root, derivatives=True)
    filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                           extension='.edf', desc='clean', preload=True)
    raw = raw_from_layout(layout, subject=subj, extension='.edf', desc=None,
                          preload=True)
    events, event_id = mne.events_from_annotations(filt)
    auds = mne.Epochs(filt, events, event_id['Audio'], baseline=None, tmin=-2,
                      tmax=5, preload=True, detrend=1)
    # bads = channel_outlier_marker(auds)
    # auds.info['bads'] = bads
