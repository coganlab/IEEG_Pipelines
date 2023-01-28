import os.path as op
from os import walk, listdir, curdir
from typing import Union, List, Tuple, Dict, Any

import mne
import re
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile
from mne_bids import read_raw_bids, BIDSPath

from Python.PreProcess.utils import PathLike, LAB_root

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
    raw = mne.io.RawArray(array * factor, info)
    return raw


def retrieve_filt(sub: str,
                  runs: Union[List[int], int] = (1, 2, 3, 4)) -> mne.io.Raw:
    """Retrieves a saved filtered fif file from the data folder."""
    try:
        iter(runs)
    except TypeError:
        runs = [runs]
    if not isinstance(runs, list):
        runs = list(runs)
    filt = mne.io.read_raw_fif(
        op.join(LAB_root, "Aaron_test", "filt_phonemesequence",
                "{}_filt_run-{}_ieeg.fif".format(sub, 1)))
    del runs[0]
    for i in runs:
        f_name = op.join(LAB_root, "Aaron_test", "filt_phonemesequence",
                         "{}_filt_run-{}_ieeg.fif".format(sub, i))
        mne.concatenate_raws([filt, mne.io.read_raw_fif(f_name)])
    return filt


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
                                "{}".format(LAB_root, task))
    sub_pad = "D" + "{}".format(sub_num).zfill(4)
    subject = "D{}".format(sub_num)
    layout = BIDSLayout(BIDS_root)
    raw = raw_from_layout(layout, sub_pad, run)
    D_dat_raw, D_dat_filt = find_dat(op.join(LAB_root, "D_Data",
                                             task, subject))
    return layout, raw, D_dat_raw, D_dat_filt


if __name__ == "__main__":
    from Python.PreProcess import utils, filter
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
    # %% Filter the data
    # filt = filter.line_filter(raw, mt_bandwidth=5.0, n_jobs=5,
    #                    filter_length='20s', verbose=10,
    #                    freqs=[60, 120, 180, 240], notch_widths=20)
    # raw.plot(n_channels=3,precompute=True, start=90)
    # filt = retrieve_filt(sub_pad, 1)
    # %% Plot the data
    #  data = [raw, filt, raw_dat, dat]
    # utl.figure_compare(data, [ "BIDS Un", "BIDS ", "Un", ""])
    # for chan in raw.ch_names:
    #     if chan == "Trigger":
    #         continue
    #     fmax = 250
    #     spectrum = raw.compute_psd(method="multitaper", fmin=0, fmax=fmax,
    #     picks=chan,
    #                                 n_jobs=cpu_count(), verbose='INFO')
    #     psds, freqs = spectrum.get_data(return_freqs=True)
