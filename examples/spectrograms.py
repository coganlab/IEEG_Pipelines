"""
Example spectrogram plot
========================

Below is a gallery of examples
"""
from PreProcess.utils import plotting
from bids import BIDSLayout
from PreProcess.utils import utils
from PreProcess.timefreq.multitaper import spectrogram
from PreProcess.navigate import crop_data, channel_outlier_marker, \
    raw_from_layout
from task.SentenceRep.events import fix_annotations
import numpy as np
import matplotlib as mpl
try:
    mpl.use("TkAgg")
except ImportError:
    pass

# %%
layout = BIDSLayout(utils.LAB_root + "/BIDS-1.0_SentenceRep/BIDS",
                    derivatives=True)
filt = raw_from_layout(layout.derivatives['clean'], subject='D0029',
                       extension='.edf', desc='clean', preload=True)

# %% Crop raw data to minimize processing time
#
# new = crop_data(filt)
new = filt.copy()

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 5)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# CAR
good.set_eeg_reference(ref_channels="average", ch_type='seeg')

# Remove intermediates from mem
del new
# %%
fix_annotations(good)
# %%
freq = np.arange(10, 200., 2.)
# resp = trial_ieeg(good, "Word/Response", (-1.5, 1.5))
spectra = spectrogram(good, freq, 'Word/Audio', -1, 1.5, 'Start', -0.5, 0,
                      n_jobs=6, verbose=10, time_bandwidth=10, n_cycles=freq/2)
# %%
# with open("spectra.npy", "rb") as f:
#     spectra = np.load(f, allow_pickle=True)[0]

plotting.chan_grid(spectra, vmin=0.7, vmax=1.4)
