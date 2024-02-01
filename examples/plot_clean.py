"""
Line noise filtering script
===================================

Filters the 60Hz line noise from the data, as well as the harmonics. Includes
environment checks for SLURM jobs for convenience
"""

import mne
import os
from ieeg.io import save_derivative, raw_from_layout
from ieeg.mt_filter import line_filter
from ieeg.viz.utils import figure_compare
from bids import BIDSLayout

# %%
# Set up paths
# ------------
HOME = os.path.expanduser("~")

# check if currently running a slurm job
# if so, set the root directory to the workspace
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 5

# %%
# Load Data
# ---------

misc_path = mne.datasets.misc.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif', preload=True)

# %%
# Filter Data
# -----------
# A filter length of 700 ms does a good job of removing 60Hz line noise, while
# a Filter length of 20000 ms does a good job of removing the harmonics (120Hz,
# 180Hz, 240Hz)
filt = line_filter(raw,
                   mt_bandwidth=10.,
                   n_jobs=6,
                   filter_length='700ms',
                   verbose=10,
                   freqs=[60],
                   notch_widths=20)

# %%
# plot the data before and after filtering
figure_compare([raw, filt],
               labels=["Un", ""],
               avg=True,
               n_jobs=6,
               verbose=10,
               proj=True,
               fmax=250)

# %%
# Save the Data
# -------------
# Save your line noise cleaned data to `bids_root`/derivatives/test folder
bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
raw = raw_from_layout(layout,
                      subject="pt1",
                      preload=True,
                      extension=".vhdr")

# Check if derivatives folder exists and create if not
if not os.path.exists(os.path.join(bids_root, "derivatives")):
    os.mkdir(os.path.join(bids_root, "derivatives"))
    os.mkdir(os.path.join(bids_root, "derivatives", "test"))
elif not os.path.exists(os.path.join(bids_root, "derivatives", "test")):
    os.mkdir(os.path.join(bids_root, "derivatives", "test"))

# save the data
save_derivative(raw, layout, "test", True)
