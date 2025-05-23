"""
Morlet Wavelet spectrogram plot
===============================

Below is a code sample for plotting wavelet spectrograms
"""
from pyqtgraph.examples.ViewBox import yScale

from ieeg.viz.ensemble import chan_grid
from bids import BIDSLayout
from ieeg.navigate import channel_outlier_marker, trial_ieeg, outliers_to_nan
from ieeg.io import raw_from_layout
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq.superlets import superlet_tfr, superlets
from ieeg.viz.parula import parula_map
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# %% First simulate some data to test superlet spectrogram
# ------------------------------------------------
# Simulate data
# -------------
# Simulate a 1D signal with some bursts of 10, 20, and 70 Hz
# and a sampling rate of 1000 Hz
fs = 1000
frange = (5, 35)
trange = (0, 10)
t = np.arange(trange[0], trange[1], 1 / fs)
burst = np.sin(2 * np.pi * 3 * t) + 1
s1 = np.sin(2 * np.pi * 10 * t) * (t > 1) * (t < 3)
s2 = np.sin(2 * np.pi * 20 * t) * (t > 4) * (t < 6)
s3 = np.sin(2 * np.pi * 30 * t) * (t > 7) * (t < 9) * burst / 2
data = s1 + s2 + s3
data = data[np.newaxis, :]

# run the superlet transform
freqs = np.linspace(frange[0], frange[1], 100)
wavelet = superlets(data, fs, freqs, 5, (10, 20))

# Create a figure and GridSpec layout
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)  # 3 rows: 1 for signal, 2 for spectrogram

# Add the signal plot (1/3 of the height)
ax_signal = fig.add_subplot(gs[0, 0])
ax_signal.plot(t, data[0], color='k')
ax_signal.set_title("Simulated data")
ax_signal.set_ylabel("Amplitude")

# Add the spectrogram plot (2/3 of the height)
ax_spectrogram = fig.add_subplot(gs[1:, 0], sharex=ax_signal)  # Spans the last 2 rows
ax_spectrogram.imshow(wavelet, aspect='auto', origin='lower',
                      extent=[trange[0], trange[1], frange[0], frange[1]],
                      cmap='jet')
ax_spectrogram.set_title("Superlet spectrogram")
ax_spectrogram.set_ylabel("Frequency (Hz)")
ax_spectrogram.set_xlabel("Time (s)")

# Hide x-axis labels for the top plot to avoid overlap
plt.setp(ax_signal.get_xticklabels(), visible=False)


# %% Try out a log scale
# Do the transform again with log scaling
freqs = np.geomspace(frange[0], frange[1], 100)
wavelet = superlets(data, fs, freqs, 1.26, (10,10))

# Create a new figure and GridSpec layout
fig_log = plt.figure(figsize=(10, 6))
gs_log = GridSpec(3, 1, figure=fig_log)  # 3 rows: 1 for signal, 2 for spectrogram

# Add the signal plot (1/3 of the height)
ax_signal_log = fig_log.add_subplot(gs[0, 0])
ax_signal_log.plot(t, data[0], color='k')
ax_signal_log.set_title("Simulated data")
ax_signal_log.set_ylabel("Amplitude")

# Add the spectrogram plot (2/3 of the height)
ax_spectrogram_log = fig_log.add_subplot(gs[1:, 0], sharex=ax_signal_log)  # Spans the last 2 rows
ax_spectrogram_log.imshow(wavelet, aspect='auto', origin='lower',
                            extent=[trange[0], trange[1], freqs[0], freqs[-1]],
                            cmap='jet')
ax_spectrogram_log.set_yscale('log')
ax_spectrogram_log.set_title("Superlet spectrogram (log scale)")
ax_spectrogram_log.set_ylabel("Frequency (Hz)")
ax_spectrogram_log.set_xlabel("Time (s)")


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
    freqs = np.geomspace(4, 500, 80)
    spec = superlet_tfr(trials, freqs, 1., (15, 15))
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
chan_grid(spec_a, vlim=(0, 20), cmap=parula_map, yscale='log')
