"""
Example line noise filtering script
===================================

Filters the 60Hz line noise from the data, as well as the harmonics. Includes
environment checks for SLURM jobs for convenience
"""

import mne
import os
from PreProcess.navigate import get_data, raw_from_layout, save_derivative, \
    open_dat_file, find_dat
from PreProcess.mt_filter import line_filter
from PreProcess.utils import plotting

# %% Set up logging
mne.set_log_file("output.log",
                 "%(levelname)s: %(message)s - %(asctime)s",
                 overwrite=True)
mne.set_log_level("INFO")
HOME = os.path.expanduser("~")

# %% check if currently running a slurm job
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 5

# %% Load Data
TASK = "SentenceRep"
sub_num = 29
subj = "D" + str(sub_num).zfill(4)
layout = get_data("SentenceRep", root=LAB_root)
raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                      preload=False)
D_dat_raw, D_dat_filt = find_dat(os.path.join(LAB_root, "D_Data", TASK,
                                              "D" + str(sub_num)))
raw_dat = open_dat_file(D_dat_raw, raw.copy().ch_names)
dat = open_dat_file(D_dat_filt, raw.copy().ch_names)

# %% filter data
filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1,
                   filter_length='700ms', verbose=10,
                   freqs=[60], notch_widths=20)
filt2 = line_filter(filt, mt_bandwidth=10., n_jobs=-1,
                    filter_length='20s', verbose=10,
                    freqs=[60, 120, 180, 240], notch_widths=20)

# %% plot the data
data = [raw, filt, raw_dat, dat]
plotting.figure_compare(data, ["BIDS Un", "BIDS ", "Un", ""], avg=True,
                        verbose=10, proj=True, fmax=250)

# %% Unfiltered
# .. image:: unfilt.png
#  :width: 400
#  :alt: Unfiltered
#

# %% Filtered
# .. image:: filt.png
#  :width: 400
#  :alt: Filtered
#

# %% Save the data
save_derivative(filt2, layout, "clean")
