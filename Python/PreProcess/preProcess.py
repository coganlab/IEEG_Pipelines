from bids import BIDSLayout
import mne
import numpy as np
from typing import Union
from mne_bids import read_raw_bids, BIDSPath

BIDS_root = r"C:\Users\ae166\Box\CoganLab\BIDS-1.0_Uniqueness_point\BIDS"


def line_filter(data: mne.io.Raw):
    #raw = mne.io.read_raw(data, preload=True)
    #raw.set_channel_types({nam: "ecog" for nam in raw.info.ch_names})
    filt = data.notch_filter(range(60, 1000, 60))
    return filt


if __name__ == "__main__":
    layout = BIDSLayout(BIDS_root)
    data = dict()
    for sub_id in layout.get_subjects():
        data[sub_id] = dict()
        for run in layout.get_runs():
            #BIDS_path = BIDSPath(subject=sub_id, run=run, suffix="ieeg",
            #                     root=BIDS_root)
            BIDS_path = BIDSPath(**layout.get_entities(subject=sub_id, run=run,
                                              suffix="ieeg"))
            raw = read_raw_bids(bids_path=BIDS_path)
            data[sub_id][run] = line_filter(raw)
