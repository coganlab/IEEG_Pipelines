"""
Multitaper spectrogram plot
===========================

Below is a code sample for plotting spectrograms
"""
from ieeg.viz import utils
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
freq = np.arange(10, 200., 4.)
spectra = spectrogram(raw, freq, 'Response', -1.5, 1.5, 'Fixation', -1.5, 0.5,
                      n_jobs=6, verbose=10, time_bandwidth=10, n_cycles=freq/2)
crop_pad(spectra, "0.5s")

# %%
# Plot data
# ---------
utils.chan_grid(spectra, vmin=0, vmax=2, cmap=parula_map)
