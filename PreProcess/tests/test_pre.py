import mne.datasets
import numpy as np
from bids import BIDSLayout
from mne.io import BaseRaw
from mne_bids import BIDSPath
from ..navigate import raw_from_layout

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
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    assert isinstance(raw, BaseRaw)


def test_line_filter():
    from ..mt_filter import line_filter
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    filt = line_filter(raw, raw.info['sfreq'], [60])
    raw_dat = raw._data
    filt_dat = filt._data
    assert filt_dat.shape == raw_dat.shape
    params = dict(method='multitaper', tmax=20, fmin=55, fmax=65,
                  bandwidth=0.5, n_jobs=8)
    rpsd = raw.compute_psd(**params)
    fpsd = filt.compute_psd(**params)
    assert np.mean(np.abs(rpsd.get_data() - fpsd.get_data())) > 1e-10


def test_spectrogram():
    from ..timefreq.multitaper import spectrogram
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    freqs = np.arange(10, 50, 2)
    spectra = spectrogram(raw, freqs, 'onset', -0.5, 1.5, 'onset', -1, -0.5,
                          n_jobs=-1, picks=[0, 1])
    out = np.mean(spectra.data, 2)
    equals = np.array(
        [[0.51684218, 0.3937984, 0.3783827, 0.79478836, 0.96084218,
          1.41922219, 1.39128511, 0.93359408, 0.68504302, 0.88633986,
          0.73573044, 0.85395559, 0.63205756, 0.97452183, 0.50505694,
          0.55386678, 0.57788044, 0.72816488, 0.99064504, 0.96370377],
         [0.60098915, 0.42650635, 0.42349106, 0.63369896, 0.68732677,
          1.16665437, 1.82588753, 1.05841983, 0.60822312, 0.55624501,
          0.48200856, 0.57212177, 0.55566463, 0.77624132, 0.64708594,
          0.97556593, 0.80279372, 0.74896282, 0.6922507, 0.73861831]])
    assert np.allclose(out, equals)
