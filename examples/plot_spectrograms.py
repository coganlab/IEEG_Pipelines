"""
Example spectrogram plot
========================

Below is a code sample for plotting spectrograms
"""
from ieeg import viz
from bids import BIDSLayout
from ieeg.timefreq.multitaper import spectrogram
from ieeg.navigate import channel_outlier_marker
from ieeg.io import raw_from_layout
import numpy as np
import mne

# %% Load Data
bids_root = mne.datasets.epilepsy_ecog.data_path()
# sample_path = mne.datasets.sample.data_path()
layout = BIDSLayout(bids_root)
filt = raw_from_layout(layout, subject="pt1", preload=True,
                       extension=".vhdr")

# %% Crop raw data to minimize processing time
#
# new = crop_data(filt)
new = filt.copy()

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 5)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# Remove intermediates from mem
del new

# %% Calculate spectra
freq = np.arange(10, 200., 2.)
spectra = spectrogram(good, freq, 'PD', -1, 1.5, 'onset', -0.5, 0,
                      n_jobs=6, verbose=10, time_bandwidth=10, n_cycles=freq/2)

# %% Plot data
viz.chan_grid(spectra, vmin=0.7, vmax=1.4)

# %% example output for our data
# .. image:: ../../examples/D29_spec.png
#  :width: 700
#  :alt: spectrogram
