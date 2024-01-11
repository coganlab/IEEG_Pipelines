"""
Line noise filtering script
===================================

Filters the 60Hz line noise from the data, as well as the harmonics. Includes
environment checks for SLURM jobs for convenience
"""

import mne
import os
from ieeg.io import raw_from_layout, save_derivative
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

bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
raw = raw_from_layout(layout, subject="pt1", preload=True,
                      extension=".vhdr")

# %%
# Filter Data
# -----------

filt = line_filter(raw, mt_bandwidth=10., n_jobs=6,
                   filter_length='700ms', verbose=10,
                   freqs=[60], notch_widths=20)

# %%
# plot the data before and after filtering
data = [raw, filt]
figure_compare(data, ["Un", ""], avg=True, n_jobs=6,
                   verbose=10, proj=True, fmax=250)

# %%
# Our Data Unfiltered
# -------------------
#
# .. image:: ../../examples/unfilt.png
#  :width: 400
#  :alt: Unfiltered
#

# %%
# Our Data Filtered
# -----------------
#
# .. image:: ../../examples/filt.png
#  :width: 400
#  :alt: Filtered
#

# %%
# Save the Data
# -------------
# to bids_root/derivatives/test folder

# Check if derivatives folder exists and create if not
if not os.path.exists(os.path.join(bids_root, "derivatives")):
    os.mkdir(os.path.join(bids_root, "derivatives"))
    os.mkdir(os.path.join(bids_root, "derivatives", "test"))
elif not os.path.exists(os.path.join(bids_root, "derivatives", "test")):
    os.mkdir(os.path.join(bids_root, "derivatives", "test"))

# save the data
save_derivative(filt, layout, "test", True)
