"""
Multitaper spectrogram plot
===========================

Below is a code sample for plotting spectrograms
"""
from ieeg.viz.ensemble import chan_grid
from ieeg.timefreq.multitaper import spectrogram
from ieeg.timefreq.utils import crop_pad
from ieeg.navigate import channel_outlier_marker
from ieeg.viz.parula import parula_map
import numpy as np
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
# Calculate Spectra
# -----------------
freq = np.arange(10, 200., 6.)
spectra = spectrogram(raw, freq, 'Response', -1.2, 1.2, 'Fixation', -1.2, 0.2,
                      n_jobs=1, verbose=10, time_bandwidth=10, n_cycles=freq/2)
crop_pad(spectra, "0.5s")

# %%
# Plot data
# ---------
chan_grid(spectra, vlim=(0, 2), cmap=parula_map)
