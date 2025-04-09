import mne
import numpy as np
import pytest
import scipy

from ieeg.calc.fast import mean_diff, ttest

bids_root = mne.datasets.epilepsy_ecog.data_path()
seeg = mne.io.read_raw(mne.datasets.misc.data_path() /
                       'seeg' / 'sample_seeg_ieeg.fif')


@pytest.mark.parametrize("mat, axis, expected", [
    # Test case 1: No mask, axis=0
    (np.array([[1, 2, 3], [4, 5, 6]]), 0, (
            [2.5, 3.5, 4.5], [1.5, 1.5, 1.5])),

    # Test case 2: No mask, axis=1
    (np.array([[1, 2, 3], [4, 5, 6]]), 1, (
            [2.0, 5.0], [0.57735027, 0.57735027])),
])
def test_dist(mat, axis, expected):
    from ieeg.calc.stats import dist  # Import your actual module here

    mean, std = dist(mat, axis)
    assert np.allclose(mean, np.array(expected[0]))  # Check mean
    assert np.allclose(std, np.array(expected[1]))  # Check standard deviation


@pytest.mark.parametrize("func, expected", [
    (mean_diff, np.arange(38, 56)),
    (scipy.stats.f_oneway, np.arange(46, 50)),
    (scipy.stats.ttest_ind, np.arange(38, 56)),
    (ttest, np.arange(38, 56))
])
def test_permclust(func, expected):
    from ieeg.navigate import trial_ieeg
    from ieeg.calc import stats

    out = []
    for epoch, t in zip(('Fixation', 'Response'), ((-0.3, 0), (-0.1, 0.2))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(seeg, epoch, times, preload=True)
        out.append(trials)
    resp = out[1]
    resp.decimate(10)
    base = out[0]
    base.decimate(10)

    mask, pvals = stats.time_perm_cluster(resp._data[:, 78],
                                          base._data[:, 78], 0.01,
                                          stat_func=func, n_perm=4000)
    assert np.mean(mask[expected]) > 0.8


def test_permclust2():
    from ieeg.calc.stats import time_perm_cluster
    base = np.load("tests/d100_basefield.npy")
    field = base[1]
    base = base[0]
    mask, pvals = time_perm_cluster(field, base, 0.1,
                                    0.1, 10000, 1, 1)
    assert np.any(mask)
    assert np.min(pvals[1]) < 0.05


def test_stats_wavelet():
    from ieeg.navigate import trial_ieeg, outliers_to_nan
    from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
    from ieeg.calc import stats

    out = []
    for epoch, t in zip(('Fixation', 'Response'), ((-0.3, 0), (-0.1, 0.2))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(seeg, epoch, times, preload=True,
                            picks=['LAMY 6', 'LAMY 7'])
        outliers_to_nan(trials, 10)
        spec = wavelet_scaleogram(trials, n_jobs=-2, decim=20)
        crop_pad(spec, "0.5s")
        out.append(spec)
    resp = out[1]
    base = out[0]

    mask, pvals = stats.time_perm_cluster(resp.data, base.data, 0.1,
                                          ignore_adjacency=1, n_perm=10000)
    mask1, pvals1 = stats.time_perm_cluster(resp.data[:, 0], base.data[:, 0],
                                            0.1, n_perm=10000)

    assert np.any(mask)
    assert np.isclose(np.mean(mask1), np.mean(mask[0]))


def test_window_averaged_shuffle():
    from ieeg.calc.stats import window_averaged_shuffle
    base = np.load("tests/d100_basefield.npy")
    field = base[1]
    base = base[0]
    p = window_averaged_shuffle(field, base, 10000, 1, 1)
    t = np.isclose(np.array([0.51315, 0.99920, 6.9993e-04]), p, 0, 0.05)
    assert np.all(t)
