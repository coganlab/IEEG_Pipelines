import mne.datasets
import numpy as np
from bids import BIDSLayout
from mne.io import BaseRaw
from mne_bids import BIDSPath


bids_root = mne.datasets.epilepsy_ecog.data_path()
# sample_path = mne.datasets.sample.data_path()
layout = BIDSLayout(bids_root)
log_filename = "output.log"
# op.join(LAB_root, "Aaron_test", "Information.log")
mne.set_log_file(log_filename,
                 "%(levelname)s: %(message)s - %(asctime)s",
                 overwrite=True)
mne.set_log_level("DEBUG")


def test_bids():
    assert "pt1" in layout.get_subjects()


def test_bidspath_from_layout():
    from ..navigate import bidspath_from_layout
    expected = "sub-pt1_ses-presurgery_task-ictal_ieeg.eeg"
    bidspath = bidspath_from_layout(layout, subject="pt1",
                                    extension=".eeg")
    assert isinstance(bidspath, BIDSPath)
    assert bidspath.basename == expected


def test_raw_from_layout():
    from ..navigate import raw_from_layout
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    assert isinstance(raw, BaseRaw)


def test_line_filter():
    from ..navigate import raw_from_layout
    from ..mt_filter import line_filter
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    raw.load_data()
    filt = line_filter(raw, raw.info['sfreq'], [60])
    raw_dat = raw._data
    filt_dat = filt._data
    assert filt_dat.shape == raw_dat.shape
    params = dict(method='multitaper', tmax=20, fmin=55, fmax=65,
                  bandwidth=0.5, n_jobs=8)
    rpsd = raw.compute_psd(**params)
    fpsd = filt.compute_psd(**params)
    assert np.mean(np.abs(rpsd.get_data()-fpsd.get_data())) > 1e-10
