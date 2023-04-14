from PreProcess.navigate import get_data, crop_data, \
    channel_outlier_marker, raw_from_layout, trial_ieeg
from PreProcess.math import stats
from PreProcess.timefreq import multitaper, utils
from mne.time_frequency import tfr_multitaper
import os.path as op
import numpy as np
import matplotlib as mpl
try:
    mpl.use("TkAgg")
except ImportError:
    pass

# %% Load the data
TASK = "SentenceRep"
sub_num = 29
subj = "D" + str(sub_num).zfill(4)
HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)
filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                       extension='.edf', desc='clean', preload=False)

# %% Crop raw data to minimize processing time
new = crop_data(filt)

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 5)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# CAR
good.set_eeg_reference(ref_channels="average", ch_type='seeg')

# Remove intermediates from mem
del new

# %% fix SentenceRep events
from task.SentenceRep.events import fix_annotations  # noqa E402
fix_annotations(good)

# %% separate events

resp = trial_ieeg(good, "Word/Response", (-1, 1))
base = trial_ieeg(good, "Start", (-0.5, 0))

# %% create spectrograms
freqs = np.arange(1, 200., 4.)
#
resp_s = tfr_multitaper(resp, freqs, n_jobs=6, verbose=10, average=False,
                        time_bandwidth=10, n_cycles=freqs/2, return_itc=False,
                        decim=20)
base_s = tfr_multitaper(base, freqs, n_jobs=6, verbose=10, average=False,
                        time_bandwidth=10, n_cycles=freqs/2, return_itc=False,
                        decim=20)

# %%
mask = stats.time_perm_cluster(resp_s.data, base_s.data, 0.05, n_perm=4)

# %%
# with open("spectra.npy", "rb") as f:
#     spectra = np.load(f, allow_pickle=True)[0]
# from PreProcess.utils import plotting
# import matplotlib
# plotting.chan_grid(spectra, vmin=0.7, vmax=1.4)
