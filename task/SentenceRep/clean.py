# %%
import mne
import os
from PreProcess.navigate import get_data, raw_from_layout, save_derivative
from PreProcess.mt_filter import line_filter

# %%
if __name__ == "__main__":
    # %% check if currently running a slurm job
    HOME = os.path.expanduser("~")
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        LAB_root = os.path.join(HOME, "workspace", "CoganLab")
        subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:  # if not then set box directory
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
        subject = 8

    # %% Load Data
    layout = get_data("SentenceRep", LAB_root)
    subjlist = layout.get_subjects()
    subjlist.sort()
    subj = subjlist[subject]
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
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
