import os.path as op
from os import walk
from re import match
from typing import Union, List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile
from joblib import cpu_count
from mne_bids import read_raw_bids, BIDSPath

if __name__ == '__main_'+'_':
    from utils import LAB_root, PathLike
else:
    from .utils import LAB_root, PathLike

RunDict = Dict[int, mne.io.Raw]
SubDict = Dict[str, RunDict]


def find_dat(folder: PathLike) -> Tuple[PathLike, PathLike]:
    """Looks for the .dat file in a specified folder"""
    cleanieeg = None
    ieeg = None
    for root, _, files in walk(folder):
        for file in files:
            if match(r".*cleanieeg\.dat.*", file):
                cleanieeg: PathLike = op.join(root, file)
            elif match(r".*ieeg\.dat.*", file):
                ieeg: PathLike = op.join(root, file)
            if ieeg is not None and cleanieeg is not None:
                return ieeg, cleanieeg
    raise FileNotFoundError("Not all .dat files were found:")


def line_filter(data: mne.io.Raw) -> mne.io.Raw:
    """Implementation of the mne notch filter. needs to be improved or replaced"""
    if not data.preload:
        data.load_data()
    filt = data.copy().notch_filter(None,  # (60, 120, 180, 240),
                                    # method='fir',
                                    # fir_window='blackman',
                                    # pad='reflect',
                                    method='spectrum_fit',
                                    mt_bandwidth=5.0,
                                    filter_length='20s',
                                    p_value=0.05,  # only used if freqs=None
                                    verbose=10,
                                    n_jobs=cpu_count())
    # make njobs 'cuda' with a gpu if method is 'fir'
    return filt


def mt_filt(data: mne.io.Raw):  # TODO: make your own filter
    """A multitaper notch filter that eliminates the strongest line

    Steps:
    1. psd_multitaper
    2. f-test pre-defined freq range for max power using mne.stats.permutation
       _cluster_test (or just max)
    3. fft extract and sum frequencies found in max
    4. subtract frequencies in the time domain (signal - extracted_signal)

    """
    f, ax = plt.subplots()
    psds, freqs = mne.time_frequency.psd_multitaper(data, fmax=250,
                                                    n_jobs=cpu_count(),
                                                    verbose=10)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0)
    psds_std = psds.std(0)
    ax.plot(freqs, psds_mean, color='k')
    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                    color='k', alpha=.5)
    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')


def bidspath_from_layout(layout: BIDSLayout, **kwargs: dict) -> BIDSPath:
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
                    extension=".edf") -> mne.io.Raw:
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
                  sfreq: int = 2048, types: str = "seeg") -> mne.io.RawArray:
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
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array, info)
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


def filt_main(layout: BIDSLayout,
              subjects: Union[str, List[str]] = None,
              runs: Union[int, List[int]] = None
              ) -> Tuple[SubDict, SubDict]:
    data = dict()
    raw_data = dict()
    subjects: List[str] = layout.get(return_type="id", target="subject",
                                     subject=subjects)
    for sub_id in subjects:
        raw_data[sub_id] = dict()
        data[sub_id] = dict()
        runs: List[int] = layout.get(return_type="id", target="run",
                                     subject=sub_id, run=runs)
        for run in runs:
            raw_data[sub_id][run] = raw_from_layout(layout, sub_id, run)
            data[sub_id][run] = line_filter(raw_data[sub_id][run])
    return raw_data, data


def filt_main_2(layout: BIDSLayout, subjects: Union[str, List[str]] = None):
    """A function that runs through all the subjects that filters then saves

    """
    if subjects is None:
        subjects: List[str] = layout.get(return_type="id", target="subject")
    for sub_id in subjects:
        runs: List[int] = layout.get(return_type="id", target="run",
                                     subject=sub_id)
        for run in runs:
            try:
                raw_data = raw_from_layout(layout, sub_id, run)
                filt_data = line_filter(raw_data)
                save_name = "{}_filt_run-{}_ieeg.fif".format(sub_id, run)
                filt_data.save(op.join(LAB_root, "Aaron_test",
                                       "filt_phonemesequence", save_name),
                               overwrite=False)
            except Exception:
                pass
        del runs


if __name__ == "__main__":
    log_filename = "output.log"
    # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    BIDS_root = op.join(LAB_root, "BIDS-1.3_Phoneme_sequencing", "BIDS")
    sub_num = 53
    sub_pad = "D00{}".format(sub_num)
    subject = "D{}".format(sub_num)
    layout = BIDSLayout(BIDS_root)
    raw = raw_from_layout(layout, sub_pad, 2)
    filt = line_filter(raw)
    # raw.plot(n_channels=3,precompute=True, start=90)
    # filt = retrieve_filt(sub_pad, 1)
    # D_dat_raw, D_dat_filt = find_dat(op.join(LAB_root, "D_Data",
    #                                 TASK, SUB))
    # raw_dat = open_dat_file(D_dat_raw, raw.copy().channels)
    # dat = open_dat_file(D_dat_filt, raw.copy().channels)
    # raw = raw_from_layout(layout, sub_pad, 1)
    # data = [raw, filt]
    # figure_compare(data, [ "BIDS Un", "BIDS "])
