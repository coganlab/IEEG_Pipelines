"""
Time Permutation Cluster Statistics
===================================

Below is a code sample for performing a permutation test with cluster
correction
"""
from ieeg.navigate import channel_outlier_marker, trial_ieeg
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc import stats
import matplotlib.pyplot as plt
import mne

# %% Load Data

misc_path = mne.datasets.misc.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')

# %% Mark channel outliers as bad
raw.info['bads'] = channel_outlier_marker(raw, 3)

# Exclude bad channels
raw.drop_channels(raw.info['bads'])
raw.load_data()

# CAR
raw.set_eeg_reference(ref_channels="average", ch_type='seeg')

# %% Run gamma filtering
out = []
for epoch, t in zip(('Fixation', 'Response'),
                    ((-0.4, 0), (-0.1, 0.3))):
    times = [None, None]
    times[0] = t[0] - 0.5
    times[1] = t[1] + 0.5
    trials = trial_ieeg(raw, epoch, times, preload=True)
    gamma.extract(trials, copy=False, n_jobs=1)
    crop_pad(trials, "0.5s")
    out.append(trials)
resp = out[1]
resp.decimate(2)
base = out[0]
base.decimate(2)

del raw

# %% run time cluster stats

mask1 = stats.time_perm_cluster(resp._data, base._data, 0.05, axis=0,
                                n_perm=1000, n_jobs=6, ignore_adjacency=1)
plt.imshow(mask1)

# %% use a windowed average test instead

# response data points are length 201, while baseline is 200. We need to
# trim the response data to match the baseline with [..., :-1]
mask2 = stats.window_averaged_shuffle(resp._data[..., :-1], base._data, 1000,
                                      stat_func=stats.mean_diff)
plt.imshow(mask2[:, None])
