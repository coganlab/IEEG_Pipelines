import logging
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

from utils import HOME, LAB_root, PathLike, figure_compare
from mri import allign_mri

BIDS_root = op.join(LAB_root, "BIDS-1.3_Phoneme_sequencing", "BIDS")
RunDict = Dict[int, mne.io.Raw]
SubDict = Dict[str, RunDict]


def find_dat(folder: PathLike) -> Tuple[PathLike, PathLike]:
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
    """

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


def get_strongest_line(array: np.ndarray):
    pass


def bidspath_from_layout(layout: BIDSLayout, **kwargs: dict) -> BIDSPath:
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
                    run: Union[List[int], int] = None) -> mne.io.Raw:
    kwargs: Dict[str, Any] = dict(subject=subject)
    if run:
        kwargs["run"] = run
    runs = layout.get(return_type="id", target="run", **kwargs)
    raw: List[mne.io.Raw] = []
    for r in runs:
        BIDS_path = bidspath_from_layout(layout, subject=subject, run=r,
                                         extension=".edf")
        new_raw = read_raw_bids(bids_path=BIDS_path)
        new_raw.load_data()
        raw.append(new_raw.copy())
        del new_raw
    whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    return whole_raw


def open_dat_file(file_path: str, channels: List[str],
                  sfreq: int = 2048, types: str = "seeg") -> mne.io.RawArray:
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")
    channels.remove("Trigger")
    array = np.reshape(data, [len(channels), -1], order='F')
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array, info)
    return raw


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
            logging.info("sub-{}, run-{}".format(sub_id, run))
            raw_data = raw_from_layout(layout, sub_id, run)
            filt_data = line_filter(raw_data)
            save_name = "{}_filt_run-{}_ieeg.fif".format(sub_id, run)
            filt_data.save(op.join(LAB_root, "Aaron_test",
                                   "filt_phonemesequence", save_name),
                           overwrite=False)
        del runs


if __name__ == "__main__":
    log_filename = op.join(LAB_root, "Aaron_test", "Information.log")
    logging.basicConfig(filename=log_filename, filemode="w",
                        level=logging.INFO)
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s")
    SUB = "D0024"
    layout = BIDSLayout(BIDS_root)
    # filt_main_2(layout)
    filt = mne.io.read_raw_fif(
        op.join(LAB_root, "Aaron_test", "filt_phonemesequence",
                "{}_filt_run-{}_ieeg.fif".format(SUB, 1)))
    for i in [2, 3, 4]:
        f_name = op.join(LAB_root, "Aaron_test", "filt_phonemesequence",
                         "{}_filt_run-{}_ieeg.fif".format(SUB, i))
        mne.concatenate_raws([filt, mne.io.read_raw_fif(f_name)])
    T1_path = layout.get(return_type="path", subject=SUB,
                         extension="nii.gz")[0]
    CT_path = T1_path.path.replace("T1w.nii.gz", "CT.nii.gz")
    gui = allign_mri(T1_path, CT_path, filt, "D24")
    # D_dat_raw, D_dat_filt = find_dat(op.join(LAB_root, "D_Data",
    #  TASK, SUB))
    # raw_dat = open_dat_file(D_dat_raw, raw.copy().channels)
    # dat = open_dat_file(D_dat_filt, raw.copy().channels)
    # data = [raw_dat, filt_dat, raw, filt]
    # figure_compare(data, ['Un',  '', "BIDS Un", "BIDS "])
