"""
Morlet Wavelet spectrogram plot
===============================

Below is a code sample for plotting wavelet spectrograms
"""
from ieeg.viz.ensemble import chan_grid
from bids import BIDSLayout
from ieeg.navigate import channel_outlier_marker, trial_ieeg, outliers_to_nan
from ieeg.io import raw_from_layout
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
from ieeg.viz.parula import parula_map
import numpy as np
import mne

# %%
# Load Data
# ---------
bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
filt = raw_from_layout(layout, subject="pt1", preload=True,
                      extension=".vhdr")

# %%
# Crop raw data to minimize processing time
# -----------------------------------------
new = filt.copy()

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 5)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# Remove intermediates from mem
del new

# %%
# Calculate Spectra
# -----------------
for epoch, t in zip(
        ("onset", "offset"),
        ((-1, 0), (-1, 1)),):
    times = [None, None]
    times[0] = t[0] - 0.5
    times[1] = t[1] + 0.5
    trials = trial_ieeg(good, epoch, times, preload=True)
    outliers_to_nan(trials, outliers=10)
    spec = wavelet_scaleogram(trials, n_jobs=1, decim=int(
        good.info['sfreq'] / 200))
    crop_pad(spec, "0.5s")
    if epoch == "onset":
        base = spec.copy()
        continue
    spec_a = rescale(spec, base, copy=True, mode='ratio').average(
        lambda x: np.nanmean(x, axis=0), copy=True)
    spec_a._data = np.log10(spec_a._data) * 20

# %%
# Plot data
# ---------
chan_grid(spec_a, vlim=(-2, 20), cmap=parula_map)
