from bids import BIDSLayout
from bids.layout import parse_file_entities, BIDSFile
import mne
import os
from joblib import Parallel, delayed
from mne_bids import read_raw_bids, BIDSPath
import numpy as np

HOME = os.path.expanduser("~")
BIDS_root = os.path.join(HOME, r"Box\CoganLab\BIDS-1.1_Uniqueness_point\BIDS")


def line_filter(data: mne.io.Raw) -> mne.io.Raw:
    if not data.preload:
        data.load_data()
    filt = data.notch_filter(freqs=range(60, 1020, 60), method='spectrum_fit',
                             verbose=10,)
    return filt


def bidspath_from_layout(layout: BIDSLayout, **kwargs) -> BIDSPath:
    my_search: list = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file, "
                                "try adding more search terms")
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    my_search: BIDSFile = my_search[0]
    entities = my_search.get_entities()
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


def raw_from_layout(layout: BIDSLayout, subject: str, run: str) -> mne.io.Raw:
    BIDS_path = bidspath_from_layout(layout, subject=subject, run=run,
                                     extension=".edf")
    raw = read_raw_bids(bids_path=BIDS_path)
    return raw


def open_dat_file(file_path: str, channels: list):
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")

    array = np.reshape(data, [len(channels), -1], order='F')
    return array


if __name__ == "__main__":
    layout = BIDSLayout(BIDS_root)
    data = dict()
    raw_data = dict()
    for sub_id in layout.get_subjects():
        if not sub_id == "D0059":
            continue
        raw_data[sub_id] = dict()
        runs = layout.get_runs()
        for run in runs:
            raw_data[sub_id][run] = raw_from_layout(layout, sub_id, run)
        filtered_runs = Parallel(n_jobs=len(runs))(delayed(
            line_filter)(raw_data[sub_id][run]) for run in runs)
        data[sub_id] = mne.concatenate_raws(filtered_runs)
        print(data)
