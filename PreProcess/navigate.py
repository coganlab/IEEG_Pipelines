import os.path as op
import re
from os import walk, listdir
from typing import Union, List, Tuple, Dict, Any

import mne
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile
from mne_bids import read_raw_bids, BIDSPath

from mt_filter import Signal, line_filter
from timefreq.utils import to_samples
from utils.utils import PathLike, LAB_root

RunDict = Dict[int, mne.io.Raw]
SubDict = Dict[str, RunDict]


def find_dat(folder: PathLike) -> Tuple[PathLike, PathLike]:
    """Looks for the .dat file in a specified folder"""
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
    """Searches a BIDSLayout for a file and returns a BIDSPath to it.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDSLayout to search.
    **kwargs : dict
        The parameters to search for. See BIDSFile.get() for more info.
    """
    my_search: List[BIDSFile] = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file, "
                                "try adding more search terms")
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    found = my_search[0]
    entities = found.get_entities()
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


def raw_from_layout(layout: BIDSLayout, subject: str,
                    run: Union[List[int], int] = None,
                    extension: str = ".edf") -> mne.io.Raw:
    """Searches a BIDSLayout for a raw file and returns a mne Raw object.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDSLayout to search.
    subject : str
        The subject to search for.
    run : Union[List[int], int], optional
        The run to search for, by default None
    extension : str, optional
        The file extension to search for, by default ".edf"
    """
    kwargs: Dict[str, Any] = dict(subject=subject)
    if run:
        kwargs["run"] = run
    runs = layout.get(return_type="id", target="run", **kwargs)
    raw: List[mne.io.Raw] = []
    if runs:
        for r in runs:
            BIDS_path = bidspath_from_layout(layout, subject=subject, run=r,
                                             extension=extension)
            new_raw = read_raw_bids(bids_path=BIDS_path)
            new_raw.load_data()
            raw.append(new_raw.copy())
            del new_raw
        whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    else:
        BIDS_path = bidspath_from_layout(layout, subject=subject,
                                         extension=extension)
        whole_raw = read_raw_bids(bids_path=BIDS_path)
    return whole_raw


def open_dat_file(file_path: str, channels: List[str],
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
    raw = mne.io.RawArray(array * factor, info, preload=False)
    return raw


def get_data(sub_num: int = 53, task: str = "SentenceRep", run: int = None,
             BIDS_root: PathLike = None, lab_root=LAB_root):
    """

    """
    for dir in listdir(lab_root):
        if re.match(r"BIDS-\d\.\d_" + task, dir) and "BIDS" in listdir(op.join(
                lab_root, dir)):
            BIDS_root = op.join(lab_root, dir, "BIDS")
            break
    if BIDS_root is None:
        raise FileNotFoundError("Could not find BIDS directory in {} for task "
                                "{}".format(lab_root, task))
    sub_pad = "D" + "{}".format(sub_num).zfill(4)
    subject = "D{}".format(sub_num)
    layout = BIDSLayout(BIDS_root)
    raw = raw_from_layout(layout, sub_pad, run)
    D_dat_raw, D_dat_filt = find_dat(op.join(lab_root, "D_Data",
                                             task, subject))
    return layout, raw, D_dat_raw, D_dat_filt


def crop_data(raw: mne.io.Raw, start_pad: str = "10s", end_pad: str = "10s"):
    '''
    Takes raw file with annotated events and crop the file so that the raw
    file starts at the first event and stops an amount of time in seconds
    given by end_pad after the last event
    '''

    start_pad = to_samples(start_pad, raw.info['sfreq']) / raw.info['sfreq']
    end_pad = to_samples(end_pad, raw.info['sfreq']) / raw.info['sfreq']

    # get start and stop time from raw.annotations onset attribute
    t_min = raw.annotations.onset[0] - start_pad
    t_max = raw.annotations.onset[-1] + end_pad

    # create new cropped raw file
    new_raw = raw.copy().crop(tmin=t_min, tmax=t_max)

    return new_raw


def channel_outlier_marker(input_raw: mne.io.Raw,
                           outlier_sd: int = 3) -> mne.io.Raw:
    """
    Marks a channel as 'bad' if the mean of the channel is different from
    the mean across channels by a factor of the cross channel std given by
    outlier_sd
    """
    data = input_raw.get_data()
    mu = np.mean(data)  # take the mean across all channels and time series
    sig = np.std(data)  # take standard deviation across all time series

    # Loop over each channel, calculate mean, and append channel to 'bad'
    # in input_raw if the difference in means is more than the given outlier_sd
    # factor (default is 3 standard deviations)
    for ii, ch in enumerate(input_raw.ch_names):
        mu_ch = np.mean(data[ii, :])
        if abs(mu_ch - mu) > (outlier_sd * sig):
            input_raw.info['bads'].append(ch)
    return input_raw


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
    layout, raw, D_dat_raw, D_dat_filt = get_data(sub_num, TASK)
    filt = mne.io.read_raw_fif(layout.root + "/derivatives/sub-D00" + str(
        sub_num) + "_" + TASK + "_filt_ieeg.fif")

    # Spectrograms
    freqs = np.arange(10, 150., 10.)
    events, event_id = mne.events_from_annotations(filt)
    auds = mne.Epochs(filt, events, event_id, tmin=-1, tmax=1, baseline=(
        -1., -.5))['Audio']
    mne.time_frequency.tfr_array_multitaper(auds.get_data(
        ), auds.info['sfreq'], freqs, time_bandwidth=5.0)
    # Crop raw data to minimize processing time
    new = crop_data(filt)

    # Mark channel outliers as bad
    marked = channel_outlier_marker(new)

    # Exclude bad channels
    good: Signal = marked.copy().drop_channels(marked.info['bads'])

    # CAR
    good_CAR = good.set_eeg_reference(ref_channels="average")
