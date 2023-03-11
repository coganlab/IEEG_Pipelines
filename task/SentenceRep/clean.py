"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import mne
import os
from PreProcess.navigate import get_data, raw_from_layout, save_derivative
from PreProcess.mt_filter import line_filter

if __name__ == "__main__":
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
    layout = get_data("SentenceRep", LAB_root)
    subj = layout.get_subjects().sort()[subject]
    raw = raw_from_layout(layout, subject=subj, extension=".edf",
                          preload=False)

    # %% filter data
    filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1,
                       filter_length='700ms', verbose=10,
                       freqs=[60], notch_widths=20)
    filt2 = line_filter(filt, mt_bandwidth=10., n_jobs=-1,
                        filter_length='20s', verbose=10,
                        freqs=[60, 120, 180, 240], notch_widths=20)
    # %% Save the data
    save_derivative(filt2, layout, "clean")
