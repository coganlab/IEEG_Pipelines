import mne
from mne.utils import verbose, fill_doc
import numpy as np
from bids import BIDSLayout
from scipy.signal import detrend

from ieeg.timefreq.utils import to_samples
from ieeg.calc import scaling, stats
from ieeg import Doubles, Signal


def crop_empty_data(raw: mne.io.Raw, start_pad: str = "10s",
                    end_pad: str = "10s") -> mne.io.Raw:
    """Crops out long stretches of data with no events.

    Takes raw instance with annotated events and crops the instance so that the
    raw file starts at start_pad before the first event and stops an amount of
    time in seconds given by end_pad after the last event.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw file to crop.
    start_pad : str, optional
        The amount of time to pad the start of the file, by default "10s"
    end_pad : str, optional
        The amount of time to pad the end of the file, by default "10s"

    Returns
    -------
    mne.io.Raw
        The cropped raw file.
    """

    crop_list = []

    start_pad = to_samples(start_pad, raw.info['sfreq']) / raw.info['sfreq']
    end_pad = to_samples(end_pad, raw.info['sfreq']) / raw.info['sfreq']

    # split annotations into blocks
    annot = raw.annotations.copy()
    block_idx = [idx + 1 for idx, val in
                 enumerate(annot) if 'BAD boundary' in val['description']]
    block_annot = [annot[i: j] for i, j in
                   zip([0] + block_idx, block_idx +
                       ([len(annot)] if block_idx[-1] != len(annot) else []))]

    for block_an in block_annot:
        # remove boundary events from annotations
        no_bound = None
        for an in block_an:
            if 'boundary' not in an['description']:
                if no_bound is None:
                    no_bound = mne.Annotations(**an)
                else:
                    an.pop('orig_time')
                    no_bound.append(**an)

        # Skip if block is all boundary events
        if no_bound is None:
            continue
        # get start and stop time from raw.annotations onset attribute
        t_min = no_bound.onset[0] - start_pad
        t_max = no_bound.onset[-1] + end_pad

        # create new cropped raw file
        crop_list.append(raw.copy().crop(tmin=t_min, tmax=t_max))

    return mne.concatenate_raws(crop_list)


@fill_doc
@verbose
def channel_outlier_marker(input_raw: Signal, outlier_sd: float = 3,
                           max_rounds: int = np.inf, axis: int = 0,
                           verbose: bool = True) -> list[str]:
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
    %(verbose)s

    Returns
    -------
    list[str]
        List of bad channel names.
    """

    data = input_raw.get_data('data')  # (trials X) channels X time
    names = input_raw.copy().pick('data').ch_names
    bads = []  # output for bad channel names

    # Pop out names to bads output using comprehension list
    for ind, i in stats.outlier_repeat(data, outlier_sd, max_rounds, axis):
        bads.append(names[ind])
        # log channels excluded per round
        if verbose:
            mne.utils.logger.info(f'outlier round {i} channels: {bads}')

    return bads


@fill_doc
@verbose
def trial_ieeg(raw: mne.io.Raw, event: str, times: Doubles,
               baseline: str = None, basetimes: Doubles = None,
               mode: str = "mean", outliers: int = None, verbose=None,
               **kwargs) -> mne.Epochs:
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
    baseline : str
        The event to epoch the baseline.
    basetimes : tuple[float, float]
        The time window to epoch around the baseline event.
    mode : str
        The mode to use for baseline rescaling. See `mne.baseline.rescale` for
        more information.
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
    """

    # determine the events
    events, ids = mne.events_from_annotations(raw)
    dat_ids = [ids[i] for i in mne.event.match_event_names(ids, event)]
    if len(dat_ids) > 1:
        event_ids = {key.replace(event, "").strip("/"): value for key, value in
                     ids.items() if value in dat_ids}
    else:
        event_ids = {key: value for key, value in ids.items() if value in
                     dat_ids}
    # epoch the data

    if baseline is None:
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=times[0],
                            tmax=times[1], baseline=None, verbose=verbose,
                            **kwargs)
    elif basetimes is None:
        raise ValueError("Baseline event input {} must be paired with times"
                         "".format(baseline))
    else:
        kwargs['preload'] = True
        epochs = trial_ieeg(raw, event, times, **kwargs)
        base = trial_ieeg(raw, baseline, basetimes, **kwargs)
        scaling.rescale(epochs, base, mode=mode, copy=False)

    if outliers is not None:
        data = detrend(epochs.get_data(), axis=-1, type="linear")
        max = np.max(np.abs(data), axis=-1)
        std = np.std(data, axis=-1)
        reject = np.any(max > (outliers * std), axis=-1)
        epochs.drop(reject, reason="outlier")

    return epochs


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
