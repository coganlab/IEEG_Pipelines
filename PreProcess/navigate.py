import os.path as op
import re
from os import walk, listdir, mkdir

import mne
from mne.utils import verbose, fill_doc
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile, parse_file_entities
from mne_bids import read_raw_bids, BIDSPath, write_raw_bids
from scipy.signal import detrend

import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from PreProcess.timefreq.utils import to_samples, Signal  # noqa: E402
from PreProcess.math.scaling import rescale  # noqa: E402
from PreProcess.utils.utils import PathLike, LAB_root  # noqa: E402

RunDict = dict[int, mne.io.Raw]
SubDict = dict[str, RunDict]
Doubles = tuple[float, float] | list[float, float] | np.ndarray


def find_dat(folder: PathLike) -> (PathLike, PathLike):
    """Looks for the .dat file in a specified folder

    Parameters
    ----------
    folder : PathLike
        The folder to search in.

    Returns
    -------
    Tuple[PathLike, PathLike]
        The paths to the ieeg and cleanieeg files.
    """
    cleanieeg = None
    ieeg = None
    for root, _, files in walk(folder):
        for file in files:
            if re.match(r".*cleanieeg\.dat.*", file):
                cleanieeg: PathLike = op.join(root, file)
            elif re.match(r".*ieeg\.dat.*", file):
                ieeg: PathLike = op.join(root, file)
            if ieeg is not None and cleanieeg is not None:
                return ieeg, cleanieeg
    raise FileNotFoundError("Not all .dat files were found:")


def bidspath_from_layout(layout: BIDSLayout, **kwargs) -> BIDSPath:
    """Searches a :class:`BIDSLayout` for a file and returns a
    :class:`BIDSPath` to it.

    Parameters
    ----------
    layout : BIDSLayout
        The :class:`BIDSLayout` to search.
    **kwargs : dict
        The parameters passed to :func:`BIDSLayout.get`

    Returns
    -------
    BIDS_path: BIDSPath
        The BIDSPath to the file.
    """
    my_search: list[BIDSFile] = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file: \n"
                                "{} \n try adding more search terms"
                                "".format(my_search))
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    found = my_search[0]
    entities = found.get_entities()
    if 'desc' in entities:
        entities['description'] = entities.pop('desc')
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


@fill_doc
def raw_from_layout(layout: BIDSLayout, preload: bool = True,
                    run: list[int] | int = None, **kwargs) -> mne.io.Raw:
    """Searches a BIDSLayout for a raw file and returns a mne Raw object.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDSLayout to search.
    %(preload)s
    run : Union[List[int], int], optional
        The run to search for, by default None
    **kwargs : dict
        The parameters passed to bids.BIDSLayout.get()

    Returns
    -------
    mne.io.Raw
    """
    if run is None:
        runs = layout.get(return_type="id", target="run", **kwargs)
    else:
        runs = list(run)
    raw: list[mne.io.Raw] = []
    if runs:
        for r in runs:
            BIDS_path = bidspath_from_layout(layout, run=r, **kwargs)
            new_raw = read_raw_bids(bids_path=BIDS_path)
            raw.append(new_raw.copy())
        whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    else:
        BIDS_path = bidspath_from_layout(layout, **kwargs)
        whole_raw = read_raw_bids(bids_path=BIDS_path)
    if preload:
        whole_raw.load_data()
    return whole_raw


def open_dat_file(file_path: str, channels: list[str],
                  sfreq: int = 2048, types: str = "seeg",
                  units: str = "uV") -> mne.io.RawArray:
    """Opens a .dat file and returns a mne.io.RawArray object.

    Parameters
    ----------
    file_path : str
        The path to the .dat file.
    channels : List[str]
        The channels to load.
    sfreq : int, optional
        The sampling frequency, by default 2048
    types : str, optional
        The channel types, by default "seeg"
    units : str, optional
        The units of the data, by default "uV"

    Returns
    -------
    mne.io.RawArray
    """
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")
    channels.remove("Trigger")
    array = np.reshape(data, [len(channels), -1], order='F')
    match units:
        case "V":
            factor = 1
        case "mV":
            factor = 1e-3
        case "uV":
            factor = 1e-6
        case "nV":
            factor = 1e-9
        case _:
            raise NotImplementedError("Unit " + units + " not implemented yet")
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array * factor, info)
    return raw


def get_data(task: str = "SentenceRep", root: str = LAB_root) -> BIDSLayout:
    """Gets the data for a subject and task.

    Parameters
    ----------
    task : str, optional
        The task to get the data for, by default "SentenceRep"
    root : PathLike, optional
        The path to the lab directory, by default LAB_root

    Returns
    -------
    layout : BIDSLayout
        The BIDSLayout for the subject.
    """
    BIDS_root = None
    for dir in listdir(root):
        if re.match(r"BIDS-\d\.\d_" + task, dir) and "BIDS" in listdir(op.join(
                root, dir)):
            BIDS_root = op.join(root, dir, "BIDS")
            break
    if BIDS_root is None:
        raise FileNotFoundError("Could not find BIDS directory in {} for task "
                                "{}".format(root, task))
    layout = BIDSLayout(BIDS_root, derivatives=True)
    return layout


def crop_data(raw: mne.io.Raw, start_pad: str = "10s", end_pad: str = "10s"
              ) -> mne.io.Raw:
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
                           max_rounds: int = np.inf, verbose: bool = True
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
    %(verbose)s

    Returns
    -------
    list[str]
        List of bad channel names.
    """

    data = input_raw.get_data('data')  # (trials X) channels X time
    names = input_raw.copy().pick('data').ch_names
    bads = []  # output for bad channel names

    # Square the data and set zeros to small positive number
    R2 = np.square(data)
    R2[np.where(R2 == 0)] = 1e-9
    ch_dim = range(len(data.shape))[-2]  # dimension corresponding to channels

    # find all axes that are not channels (example: time, trials)
    axes = tuple(i for i in range(len(data.shape)) if not i == ch_dim)

    # Initialize stats loop
    sig = np.std(R2, axes)  # take standard deviation of each channel
    cutoff = (outlier_sd * np.std(sig)) + np.mean(sig)  # outlier cutoff
    i = 1

    # remove bad channels and re-calculate variance until no outliers are left
    while np.any(np.where(sig > cutoff)) and i <= max_rounds:

        # Pop out names to bads output using comprehension list
        [bads.append(names.pop(out-j)) for j, out in enumerate(
            np.where(sig > cutoff)[0])]

        # log channels excluded per round
        if verbose:
            mne.utils.logger.info(f'outlier round {i} channels: {bads}')

        # re-calculate per channel variance
        R2 = R2[..., np.where(sig < cutoff)[0], :]
        sig = np.std(R2, axes)
        cutoff = (outlier_sd * np.std(sig)) + np.mean(sig)
        i += 1

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
    event_ids = {key.replace(event, "").strip("/"): value for key, value in
                 ids.items() if value in dat_ids}
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
        rescale(epochs, base, mode=mode, copy=False)

    if outliers is not None:
        data = detrend(epochs.get_data(), axis=-1, type="linear")
        max = np.max(np.abs(data), axis=-1)
        std = np.std(data, axis=-1)
        reject = np.any(max > (outliers * std), axis=-1)
        epochs.drop(reject, reason="outlier")

    return epochs


@fill_doc
@verbose
def save_derivative(inst: Signal, layout: BIDSLayout, pipeline: str,
                    overwrite: bool = False, verbose=None):
    """Save an intermediate data instance from a pipeline to a BIDS folder.

    Parameters
    ----------
    inst : Signal
        The data instance to save.
    layout : BIDSLayout
        The BIDSLayout of the original data.
    pipeline : str
        The name of the pipeline.
    %(overwrite)s
    %(verbose)s
    """
    save_dir = op.join(layout.root, "derivatives", pipeline)
    if not op.isdir(save_dir):
        mkdir(save_dir)
    bounds = inst.annotations.copy()
    bounds = bounds[np.where(bounds.description == 'BAD boundary')[0]]
    bounds = [0] + list(bounds.onset) + [inst.times[-1]]
    for i, file in enumerate(inst.filenames):
        entities = parse_file_entities(file)
        entities['description'] = pipeline
        bids_path = BIDSPath(**entities, root=save_dir)
        run = inst.copy().crop(tmin=bounds[i], tmax=bounds[i+1])
        write_raw_bids(run, bids_path, allow_preload=True, format='EDF',
                       acpc_aligned=True, overwrite=overwrite, verbose=verbose)


if __name__ == "__main__":
    # %% Set up logging
    log_filename = "output.log"
    # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
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
