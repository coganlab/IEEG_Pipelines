"""
Example of Time Permutation Cluster Statistics
===================================

Below is a code sample for performing a permutation test with cluster
correction
"""
from ieeg.navigate import channel_outlier_marker, trial_ieeg
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc import stats
import matplotlib as mpl
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

# import scipy
mask = stats.time_perm_cluster(resp._data, base._data, 0.05,
                               # stat_func=scipy.stats.ttest_ind,
                               n_perm=1000)
mpl.pyplot.imshow(mask)
