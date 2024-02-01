"""
Time and Frequency Permutation Cluster Statistics
=================================================

Below is a code sample for performing a permutation test with cluster
correction
"""
from ieeg.navigate import trial_ieeg, outliers_to_nan
from ieeg.calc import stats
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import matplotlib.pyplot as plt
import mne

# %%
# Load and Rereference Data
# -------------------------

misc_path = mne.datasets.misc.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')

# Exclude bad channels
raw.load_data()

# CAR (common average reference)
raw.set_eeg_reference(ref_channels="average", ch_type='seeg')

# %%
# Calculate Spectra
# -----------------
out = []
for epoch, t in zip(('Fixation', 'Response'),  # epochs to extract
                    ((-0.3, 0), (-0.1, 0.2))):  # times to extract
    times = [None, None]
    times[0] = t[0] - 0.5  # add 0.5s to the beginning
    times[1] = t[1] + 0.5  # add 0.5s to the end
    trials = trial_ieeg(raw, epoch, times, preload=True)
    # values greater than 10 standard deviations from the mean are set to NaN
    outliers_to_nan(trials, 10)
    spec = wavelet_scaleogram(trials,
                              n_jobs=-2,
                              decim=20)
    # trim 0.5 seconds on the beginning and end of the data (edge artifacts)
    crop_pad(spec, "0.5s")
    out.append(spec)
resp = out[1]
base = out[0]

# %%
# Time Cluster Statistics
# -----------------------

mask = stats.time_perm_cluster(resp._data, base._data,
                               p_thresh=0.1,
                               ignore_adjacency=1,  # ignore channel adjacency
                               n_perm=2000)

# %%
# Plot the Time-Frequency Clusters
# --------------------------------
fig, axs = plt.subplots(5, 24, figsize=(20, 20))
for i, ax in enumerate(axs.flat):
    if i >= mask.shape[0]:
        ax.axis('off')
        continue
    ax.imshow(mask[i])
    ax.set_title(resp.info['ch_names'][i])
