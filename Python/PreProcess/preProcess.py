from bids import BIDSLayout
from bids.layout import parse_file_entities, BIDSFile
import mne
import os
from typing import Union
from mne_bids import read_raw_bids, BIDSPath

HOME = os.path.expanduser("~")
BIDS_root = os.path.join(HOME, r"Box\CoganLab\BIDS-1.0_Uniqueness_point\BIDS")


def line_filter(data: mne.io.Raw) -> mne.io.Raw:
    if not data.preload:
        data.load_data()
    filt = data.notch_filter(range(60, 1000, 60))
    return filt


def bidspath_from_layout(layout: BIDSLayout, **kwargs) -> BIDSPath:
    my_search: list = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise IndexError("Search terms matched more than one file, try adding"
                         " more search terms")
    elif len(my_search) == 0:
        raise IndexError("No files match your search terms")
    my_search: BIDSFile = my_search[0]
    entities = parse_file_entities(my_search.relpath)
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


if __name__ == "__main__":
    layout = BIDSLayout(BIDS_root)
    data = dict()
    for sub_id in layout.get_subjects():
        data[sub_id] = dict()
        for run in layout.get_runs():
            BIDS_path = bidspath_from_layout(layout, subject=sub_id, run=run,
                                             extension=".edf")
            raw = read_raw_bids(bids_path=BIDS_path)
            data[sub_id][run] = line_filter(raw)
