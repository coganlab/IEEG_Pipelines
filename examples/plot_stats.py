"""
Time Permutation Cluster Statistics
===================================

Below is a code sample for performing a permutation test with cluster
correction, as well as using the window averaged test.
"""
from ieeg.navigate import channel_outlier_marker, trial_ieeg
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc import stats
import matplotlib.pyplot as plt
import mne

# %%
# Load Data
# ---------

misc_path = mne.datasets.misc.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')

# %%
# Remove bad channels
# -------------------
# mark channel outliers as bad
raw.info['bads'] = channel_outlier_marker(raw, 3)

# Exclude bad channels
raw.drop_channels(raw.info['bads'])
raw.load_data()

# CAR (common average reference)
raw.set_eeg_reference(ref_channels="average", ch_type='seeg')

# %%
# Gamma Filtering
# ---------------
out = []
for epoch, t in zip(('Fixation', 'Response'),
                    ((-0.4, 0), (-0.1, 0.3))):
    times = [None, None]
    times[0] = t[0] - 0.5  # add 0.5s to the beginning
    times[1] = t[1] + 0.5  # add 0.5s to the end
    trials = trial_ieeg(raw, epoch, times, preload=True)
    gamma.extract(trials, copy=False, n_jobs=1)
    # trim 0.5 seconds on the beginning and end of the data (edge artifacts)
    crop_pad(trials, "0.5s")
    out.append(trials)
resp = out[1]
resp.decimate(2)
base = out[0]
base.decimate(2)

del raw

# %%
# Time Cluster Stats
# ------------------

mask1, _ = stats.time_perm_cluster(resp._data, base._data,
                                p_thresh=0.05,
                                axis=0,
                                n_perm=1000,
                                n_jobs=1,
                                ignore_adjacency=1)
fig1 = plt.imshow(mask1)

# %%
# Windowed Average Test (alternative test)
# ----------------------------------------

# response data points are length 201, while baseline is 200. We need to
# trim the response data to match the baseline with [..., :-1]
mask2 = stats.window_averaged_shuffle(resp._data[..., :-1], base._data,
                                      n_perm=1000,
                                      stat_func=stats.mean_diff)
fig2 = plt.imshow(0.05 >= mask2[:, None])
