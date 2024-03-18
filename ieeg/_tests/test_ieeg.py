import os

import mne
import numpy as np
import pytest
from bids import BIDSLayout
from mne.io import BaseRaw
from mne_bids import BIDSPath

from ieeg.io import raw_from_layout

bids_root = mne.datasets.epilepsy_ecog.data_path()
seeg = mne.io.read_raw(mne.datasets.misc.data_path() /
                       'seeg' / 'sample_seeg_ieeg.fif')
layout = BIDSLayout(bids_root)


def test_bids():
    assert "pt1" in layout.get_subjects()


def test_bidspath_from_layout():
    from ieeg.io import bidspath_from_layout
    expected = "sub-pt1_ses-presurgery_task-ictal_ieeg.eeg"
    bidspath = bidspath_from_layout(layout, subject="pt1",
                                    extension=".eeg")
    assert isinstance(bidspath, BIDSPath)
    assert bidspath.basename == expected


def test_raw_from_layout():
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    assert isinstance(raw, BaseRaw)


@pytest.mark.parametrize("n_jobs", [1, 8])
def test_line_filter(n_jobs):
    from ieeg.mt_filter import line_filter
    mne.set_log_file("output.log",
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("DEBUG")
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    filt = line_filter(raw, raw.info['sfreq'], [60])
    raw_dat = raw._data
    filt_dat = filt._data
    assert filt_dat.shape == raw_dat.shape
    params = dict(method='multitaper', tmax=20, fmin=55, fmax=65,
                  bandwidth=0.5, n_jobs=n_jobs)
    rpsd = raw.compute_psd(**params)
    fpsd = filt.compute_psd(**params)
    assert np.mean(np.abs(rpsd.get_data() - fpsd.get_data())) > 1e-10


if os.path.isfile("spec.npy"):
    spec_check = np.load("spec.npy")
else:
    spec_check = np.load("ieeg/_tests/spec.npy")


def test_spect_1():
    from ieeg.timefreq.multitaper import spectrogram
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    freqs = np.arange(10, 20, 2)
    spectra = spectrogram(raw, freqs, 'onset', -0.5, 0.5, 'onset', -1, -0.5,
                          n_jobs=-1, picks=[0, 1], decim=10, pad="0.5s")
    out = spectra._data
    assert np.allclose(out, spec_check)


def test_spect_2():
    from ieeg.timefreq.multitaper import spectrogram
    from ieeg.navigate import trial_ieeg
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    freqs = np.arange(10, 20, 2)
    on1 = trial_ieeg(raw, 'onset', (-1, 1))
    on2 = trial_ieeg(raw, 'onset', (-1.5, 0))
    spectra = spectrogram(on1, freqs, on2, n_jobs=-1, picks=[0, 1], decim=10,
                          pad="0.5s")
    out = spectra._data
    assert np.allclose(out, spec_check)


@pytest.mark.parametrize("input1, input2, expected", [
    (4, np.inf, ['LAMY 7', 'RAHP 3']),
    (3, 2, ['LAMY 7', 'LPHG 6', 'LBRI 3', 'RAHP 3', 'LENT 3', 'LPIT 5'])
])
def test_outlier(input1, input2, expected):
    from ieeg.navigate import channel_outlier_marker
    outs = channel_outlier_marker(seeg, input1, input2)
    assert outs == expected


@pytest.mark.parametrize("outliers, n_out", [
    (4, 89955),
    (5, 25987)
])
def test_trial_outlier(outliers, n_out):
    from ieeg.navigate import trial_ieeg, outliers_to_nan
    trials = trial_ieeg(seeg, 'Response', (-1, 1))
    outs = outliers_to_nan(trials, outliers)
    assert np.isnan(outs._data).sum() == n_out
