from PreProcess.navigate import get_data, save_derivative
from PreProcess.mt_filter import line_filter
import os.path as op
import mne
import os

# %% Set up logging
mne.set_log_file("output.log",
                 "%(levelname)s: %(message)s - %(asctime)s",
                 overwrite=True)
mne.set_log_level("INFO")

if __name__ == "__main__":
    TASK = "Neighborhood_Sternberg"
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        sub_num = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        sub_num = 29
    HOME = op.expanduser("~")
    LAB_root = op.join(HOME, "Box", "CoganLab")
    layout, raw = get_data(sub_num, TASK, lab_root=LAB_root)
    # %% filter data
    filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1,
                       filter_length='700ms', verbose=10,
                       freqs=[60], notch_widths=20)
    filt2 = line_filter(filt, mt_bandwidth=10., n_jobs=-1,
                        filter_length='20s', verbose=10,
                        freqs=[60, 120, 180, 240], notch_widths=20)
    # %% Save the data
    save_derivative(filt2, layout, "clean")
