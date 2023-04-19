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
conds = ["resp", "aud", "go"]

# %% Load the data
for subj in layout.get_subjects():
    subject = "D" + str(subj).zfill(4)
    for cond in conds:
        epochs = mne.read_epochs(os.path.join(
            layout.derivatives['stats'], f"/{subject}_{cond}_power-epo.fif"))
