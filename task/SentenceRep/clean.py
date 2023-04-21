# %%
import os
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.mt_filter import line_filter

# %%
if __name__ == "__main__":
    # %% check if currently running a slurm job
    HOME = os.path.expanduser("~")
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        LAB_root = os.path.join(HOME, "workspace", "CoganLab")
        subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:  # if not then set box directory
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
        subject = 0

    # %% Load Data
    layout = get_data("SentenceRep", LAB_root)
    subjlist = layout.get_subjects()
    subjlist.sort()
    subj = subjlist[subject]
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                          preload=True)

    # %% filter data
    line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
                filter_length='700ms', freqs=[60], notch_widths=20)
    line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
                filter_length='20s', freqs=[60, 120, 180, 240],
                notch_widths=20)
    # %% Save the data
    save_derivative(raw, layout, "clean")
