import mne.datasets
import numpy as np
from bids import BIDSLayout
from mne.io import BaseRaw
from mne_bids import BIDSPath
from PreProcess.navigate import raw_from_layout

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
    from PreProcess.navigate import bidspath_from_layout
    expected = "sub-pt1_ses-presurgery_task-ictal_ieeg.eeg"
    bidspath = bidspath_from_layout(layout, subject="pt1",
                                    extension=".eeg")
    assert isinstance(bidspath, BIDSPath)
    assert bidspath.basename == expected


def test_raw_from_layout():
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    assert isinstance(raw, BaseRaw)


def test_line_filter():
    from PreProcess.mt_filter import line_filter
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


spec_check = np.load("PreProcess/_tests/spec.npy")


def test_spect_1():
    from PreProcess.timefreq.multitaper import spectrogram
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    freqs = np.arange(10, 20, 2)
    spectra = spectrogram(raw, freqs, 'onset', -0.5, 0.5, 'onset', -1, -0.5,
                          n_jobs=-1, picks=[0, 1], decim=10, pad="0.5s")
    out = spectra._data
    assert np.allclose(out, spec_check)


def test_spect_2():
    from PreProcess.timefreq.multitaper import spectrogram
    from PreProcess.navigate import trial_ieeg
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    freqs = np.arange(10, 20, 2)
    on1 = trial_ieeg(raw, 'onset', (-1.5, 0))
    on2 = trial_ieeg(raw, 'onset', (-1, 1))
    spectra = spectrogram(on1, freqs, on2, n_jobs=-1, picks=[0, 1], decim=10,
                          pad="0.5s")
    out2 = spectra._data
    assert np.allclose(out, spec_check)
