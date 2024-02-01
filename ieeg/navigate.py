import mne
import numpy as np
from bids import BIDSLayout
from mne.utils import fill_doc, verbose
from scipy.signal import detrend

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

    tmp = input_raw.copy()
    data = detrend(tmp.get_data('data'))  # channels X time
    names = tmp.pick('data').ch_names
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
        if not hasattr(tmp, 'filenames'):
            raise ValueError("Raw instance must have filenames attribute to "
                             "save bad channels")
        for file in tmp.filenames:
            update(file, bads, desc)

    return bads


@verbose
def outliers_to_nan(trials: mne.epochs.BaseEpochs, outliers: float,
                    copy: bool = False, picks: list = 'data',
                    verbose=None) -> mne.epochs.BaseEpochs:
    """Set outliers to nan.

    Parameters
    ----------
    trials : mne.epochs.BaseEpochs
        The trials to remove outliers from.
    outliers : float
        The number of standard deviations above the mean to be considered an
        outlier.
    copy : bool, optional
        Whether to copy the data, by default False
    picks : list, optional
        The channels to remove outliers from, by default 'data'

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
    >>> epochs = trial_ieeg(raw, "AD1-4, ATT1,2", (-1, 2), preload=True,
    ... verbose=False)
    >>> epochs = outliers_to_nan(epochs, 3, verbose=False)
    >>> epochs['AD1-4, ATT1,2'].get_data()[0]
    array([[        nan,         nan,         nan, ...,         nan,
                    nan,         nan],
           [-0.00030586, -0.00030625, -0.00031171, ..., -0.00016054,
            -0.00015976, -0.00015664],
           [        nan,         nan,         nan, ...,         nan,
                    nan,         nan],
           ...,
           [-0.00021483, -0.00021131, -0.00023084, ..., -0.00034295,
            -0.00032381, -0.00031444],
           [-0.00052188, -0.00052852, -0.00053125, ..., -0.00046211,
            -0.00047148, -0.00047891],
           [-0.00033708, -0.00028005, -0.00020934, ..., -0.00040934,
            -0.00042341, -0.00040973]])
    """
    if copy:
        trials = trials.copy()
    picks = mne.io.pick._picks_to_idx(trials.info, picks)
    trials.load_data()
    data = trials.get_data(picks=picks, verbose=verbose)

    # bool array of where to keep data trials X channels
    keep = stats.find_outliers(data, outliers)

    # set outliers to nan if not keep
    data = np.where(keep[..., None], data, np.nan)
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
    bids_root = LAB_root + "/BIDS-1.0_SentenceRep/BIDS"
    layout = BIDSLayout(bids_root, derivatives=True)
    filt = raw_from_layout(layout.derivatives['filt'], subject=subj,
                           extension='.edf', desc='filt', preload=True)
    raw = raw_from_layout(layout, subject=subj, extension='.edf', desc=None,
                          preload=True)
    events, event_id = mne.events_from_annotations(filt)
    auds = mne.Epochs(filt, events, event_id['Audio'], baseline=None, tmin=-2,
                      tmax=5, preload=True, detrend=1)
    bads = channel_outlier_marker(auds)
    auds.info['bads'] = bads
