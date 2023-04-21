from ieeg.viz import chan_grid
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data
from ieeg.calc import stats
from mne.time_frequency import tfr_multitaper
import os.path as op
import numpy as np


# %% Load the data
TASK = "SentenceRep"
sub_num = 24
subj = "D" + str(sub_num).zfill(4)
HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)
filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                       extension='.edf', desc='clean', preload=False)

# %% Crop raw data to minimize processing time
new = crop_empty_data(filt)

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 4)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# CAR
good.set_eeg_reference(ref_channels="average", ch_type='ecog')

# Remove intermediates from mem
del new

# %% fix SentenceRep events
from events import fix_annotations  # noqa E402
fix_annotations(good)

# %% separate events

resp = trial_ieeg(good, "Word/Response", (-1.5, 1.5), preload=True, outliers=8)
base = trial_ieeg(good, "Start", (-1, 0.5), preload=True, outliers=8)

# %% create spectrograms
freqs = np.arange(1, 200., 3.)
#
resp_s = tfr_multitaper(resp, freqs, n_jobs=6, verbose=10, average=False,
                        time_bandwidth=10, n_cycles=freqs/2, return_itc=False,
                        decim=20)
resp_s.crop(tmin=-1, tmax=1)
base_s = tfr_multitaper(base, freqs, n_jobs=6, verbose=10, average=False,
                        time_bandwidth=10, n_cycles=freqs/2, return_itc=False,
                        decim=20)
base_s.crop(tmin=-0.5, tmax=0)

# %%
from scipy.signal import detrend
sig1 = resp_s.data
sig2 = base_s.data
sig2 = np.pad(sig2, ((0, 0), (0, 0), (0, 0), (
    0, sig1.shape[-1] - sig2.shape[-1])), mode='reflect')
mask = stats.time_perm_cluster(sig1, sig2, 0.05, n_perm=500,
                               ignore_adjacency=1)
signif = resp_s.copy().average()
signif._data = mask
signif.save(op.join(layout.root, 'derivatives', 'stats', '4d',
                    f"{subj}_resp_power-tfr.h5"))

# %%
# with open("spectra.npy", "rb") as f:
#     spectra = np.load(f, allow_pickle=True)[0]
chan_grid(signif, vmin=0, vmax=1)
