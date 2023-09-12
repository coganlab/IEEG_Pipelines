import numpy as np
import pytest
import scipy
import mne

from ieeg.calc.stats import mean_diff

bids_root = mne.datasets.epilepsy_ecog.data_path()
seeg = mne.io.read_raw(mne.datasets.misc.data_path() /
                       'seeg' / 'sample_seeg_ieeg.fif')


@pytest.mark.parametrize("mat, mask, axis, expected", [
    # Test case 1: No mask, axis=0
    (np.array([[1, 2, 3], [4, 5, 6]]), None, 0, (
            [2.5, 3.5, 4.5], [1.06066017, 1.06066017, 1.06066017])),

    # Test case 2: No mask, axis=1
    (np.array([[1, 2, 3], [4, 5, 6]]), None, 1, (
            [2.0, 5.0], [0.47140452, 0.47140452])),

    # Test case 3: With mask, axis=0
    (np.array([[1, 2, 3], [4, np.nan, 6]]), np.array([[1, 1, 1], [1, 0, 1]]),
     0, ([2.5, 2.0, 4.5], [1.06066017, 2., 1.06066017])),

    # Test case 4: With mask, axis=1
    (np.array([[1, 2, 3], [4, np.nan, 6]]), np.array([[1, 1, 1], [1, 0, 1]]),
     1, ([2.0, 5.0], [0.47140452, 2.59807621])),
])
def test_dist(mat, mask, axis, expected):
    from ieeg.calc.stats import dist  # Import your actual module here

    result = dist(mat, mask, axis)
    assert np.allclose(result[0], expected[0])  # Check mean
    assert np.allclose(result[1], expected[1])  # Check standard deviation


@pytest.mark.parametrize("func, expected", [
    (mean_diff, np.arange(38, 56)),
    (scipy.stats.f_oneway, np.arange(46, 50)),
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
    resp = out[1]
    resp.decimate(10)
    base = out[0]
    base.decimate(10)

    mask = stats.time_perm_cluster(resp.copy()._data[:, 78:79],
                                   base.copy()._data[:, 78:79], 0.01,
                                   stat_func=func, n_perm=4000)
    assert np.mean(mask[:, expected]) > 0.8


def test_stats_wavelet():
    from ieeg.navigate import trial_ieeg, outliers_to_nan
    from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
    from ieeg.calc import stats

    out = []
    for epoch, t in zip(('Fixation', 'Response'), ((-0.3, 0), (-0.1, 0.2))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(seeg, epoch, times, preload=True, picks=[78])
        outliers_to_nan(trials, 7)
        spec = wavelet_scaleogram(trials, n_jobs=-2, decim=20)
        crop_pad(spec, "0.5s")
        out.append(spec)
    resp = out[1]
    base = out[0]

    mask = stats.time_perm_cluster(resp._data, base._data, 0.05, n_perm=2000)

    assert np.all(mask[2:7, 11:12])
