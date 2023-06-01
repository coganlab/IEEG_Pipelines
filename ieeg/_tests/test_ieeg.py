import mne
import pytest
import numpy as np
from bids import BIDSLayout
from mne.io import BaseRaw
from mne_bids import BIDSPath
from ieeg.io import raw_from_layout
from ieeg.calc.stats import mean_diff
import scipy
import os

bids_root = mne.datasets.epilepsy_ecog.data_path()
seeg = mne.io.read_raw(mne.datasets.misc.data_path() /
                       'seeg' / 'sample_seeg_ieeg.fif')
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
    from ieeg.io import bidspath_from_layout
    expected = "sub-pt1_ses-presurgery_task-ictal_ieeg.eeg"
    bidspath = bidspath_from_layout(layout, subject="pt1",
                                    extension=".eeg")
    assert isinstance(bidspath, BIDSPath)
    assert bidspath.basename == expected


def test_raw_from_layout():
    raw = raw_from_layout(layout, subject="pt1", extension=".vhdr")
    assert isinstance(raw, BaseRaw)


def test_line_filter():
    from ieeg.mt_filter import line_filter
    raw = raw_from_layout(layout, subject="pt1", preload=True,
                          extension=".vhdr")
    filt = line_filter(raw, raw.info['sfreq'], [60])
    raw_dat = raw._data
    filt_dat = filt._data
    assert filt_dat.shape == raw_dat.shape
    params = dict(method='multitaper', tmax=20, fmin=55, fmax=65,
                  bandwidth=0.5, n_jobs=1)
    rpsd = raw.compute_psd(**params)
    fpsd = filt.compute_psd(**params)
    assert np.mean(np.abs(rpsd.get_data() - fpsd.get_data())) > 1e-10


@pytest.mark.parametrize("input_mat, shape, expected", [
    (np.zeros((10, 52)), (5, 104), np.zeros((5, 104))),
    (np.zeros((10, 50, 52)), (5, 50, 104), np.zeros((5, 50, 104))),
    (np.zeros((10, 50, 50)), (5, 50, 104), np.zeros((4, 50, 104))),
    (np.zeros((10, 100, 50, 52)), (5, 100, 50, 104),
     np.zeros((5, 100, 50, 104))),
    (np.zeros((10, 100, 50, 50)), (5, 100, 50, 104),
     np.zeros((4, 100, 50, 104))),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (1, 2, 4),
     np.array([[[1, 2, 5, 6], [3, 4, 7, 8]]]))
])
def test_same(input_mat, shape, expected):
    from ieeg.calc.stats import make_data_shape
    new_shape = make_data_shape(input_mat, shape)
    assert np.all(new_shape == expected)


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


@pytest.mark.parametrize("func, expected", [
    (mean_diff, np.arange(38, 56)),
    (scipy.stats.f_oneway, np.concatenate([
        np.arange(36, 42), np.arange(45, 50), np.arange(59, 62)])),
    (scipy.stats.ttest_ind, np.arange(38, 56))
])
def test_stats(func, expected):
    from ieeg.navigate import trial_ieeg
    from ieeg.calc import stats

    out = []
    for epoch, t in zip(('Fixation', 'Response'), ((-0.3, 0), (-0.1, 0.2))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(seeg, epoch, times, preload=True)
        out.append(trials)
    resp = out[1].copy()
    resp.decimate(10)
    base = out[0].copy()
    base.decimate(10)

    mask = stats.time_perm_cluster(resp.copy()._data[:, 78:79],
                                   base.copy()._data[:, 78:79], 0.01,
                                   stat_func=func, n_perm=4000)
    assert np.all(mask[:, expected])
