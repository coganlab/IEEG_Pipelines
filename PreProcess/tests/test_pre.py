from bids import BIDSLayout
from mne_bids import BIDSPath
import mne.datasets
from mne.io import BaseRaw

from Python.PreProcess import preProcess as pre  # noqa: E402

bids_root = mne.datasets.epilepsy_ecog.data_path()
# sample_path = mne.datasets.sample.data_path()
layout = BIDSLayout(bids_root)


def test_bids():
    assert "pt1" in layout.get_subjects()


def test_bidspath_from_layout():

    expected = "sub-pt1_ses-presurgery_task-ictal_ieeg.eeg"
    bidspath = pre.bidspath_from_layout(layout, subject="pt1",
                                        extension=".eeg")
    assert isinstance(bidspath, BIDSPath)
    assert bidspath.basename == expected


def test_raw_from_layout():
    raw = pre.raw_from_layout(layout, subject="pt1", extension=".vhdr")
    assert isinstance(raw, BaseRaw)
