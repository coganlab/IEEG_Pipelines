import mne
import os
import numpy as np
from PreProcess.navigate import get_data

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 29
layout = get_data("SentenceRep", root=LAB_root)
conds = {"resp": (-1, 1),
         "aud": (-0.5, 1.5),
         "go": (-0.5, 1.5)}

# %% Load the data
chn_names = []
epochs = dict()
signif = dict()
all_power = dict()
all_sig = dict()
for cond in conds.keys():
    all_power[cond] = np.empty((0, 200))
    all_sig[cond] = np.empty((0, 200))
folder = os.path.join(layout.root, 'derivatives', 'stats')
for subject in layout.get_subjects():
    epochs[subject] = dict()
    signif[subject] = dict()
    for cond in conds.keys():
        try:
            epochs[subject][cond] = mne.read_epochs(os.path.join(
                folder, f"{subject}_{cond}_power-epo.fif"))
            signif[subject][cond] = mne.read_evokeds(os.path.join(
                folder, f"{subject}_{cond}_mask-ave.fif"))[0]
        except FileNotFoundError:
            continue
        power = epochs[subject][cond].average()
        names = [subject + '-' + ch for ch in power.ch_names]

        # add new channels to list if not already there
        chn_names = chn_names + [ch for ch in names if ch not in chn_names]

        # add new channels to power and significance matrix
        all_power[cond] = np.vstack((all_power[cond], power.get_data()))
        all_sig[cond] = np.vstack((all_sig[cond],
                                   signif[subject][cond].get_data()))
