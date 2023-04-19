import mne
import os
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
for subject in layout.get_subjects():
    epochs[subject] = dict()
    signif[subject] = dict()
    for cond in conds.keys():
        folder = os.path.join(layout.root, 'derivatives', 'stats')
        try:
            epochs[subject][cond] = mne.read_epochs(os.path.join(
                folder, f"{subject}_{cond}_power-epo.fif"))
            signif[subject][cond] = mne.read_evokeds(os.path.join(
                folder, f"{subject}_{cond}_mask-ave.fif"))
        except FileNotFoundError:
            continue
        power = epochs[subject][cond].average()
        names = [subject + '-' + ch for ch in power.ch_names]
        chn_names = chn_names + [names[names not in chn_names]]
        # all_power = all_power + power.data
