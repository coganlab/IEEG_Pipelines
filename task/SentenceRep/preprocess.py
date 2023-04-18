import matplotlib as mpl
try:
    mpl.use("TkAgg")
except ImportError:
    pass
from PreProcess.navigate import get_data, crop_data, \
    channel_outlier_marker, raw_from_layout, trial_ieeg
from PreProcess.timefreq import gamma, utils
from PreProcess.math import scaling, stats
import os.path as op
import os

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 29

# %% Load the data
TASK = "SentenceRep"
subj = "D" + str(subject).zfill(4)
HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)
filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                       extension='.edf', desc='clean', preload=False)

# %% Crop raw data to minimize processing time
new = crop_data(filt)

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 5)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# CAR
good.set_eeg_reference(ref_channels="average", ch_type='seeg')

# Remove intermediates from mem
del new

# %% fix SentenceRep events
from task.SentenceRep.events import fix_annotations  # noqa E402
fix_annotations(good)

# %% High Gamma Filter and epoching
out = []

for epoch, t in zip(("Start", "Word/Response", "Word/Audio", "Word/Speak"),
                    ((-0.5, 0), (-1, 1), (-0.5, 1.5), (-0.5, 1.5))):
    times = [None, None]
    times[0] = t[0] - 0.5
    times[1] = t[1] + 0.5
    trials = trial_ieeg(good, epoch, times, preload=True)
    gamma.extract(trials, copy=False, n_jobs=1)
    utils.crop_pad(trials, "0.5s")
    trials.decimate(20)
    trials.filenames = good.filenames
    out.append(trials)
resp = scaling.rescale(out[1], out[0], 'mean', True)
aud = scaling.rescale(out[2], out[0], 'mean', True)
go = scaling.rescale(out[3], out[0], 'mean', True)
base = out[0]
# %% run time cluster stats
import mne
save_dir = op.join(layout.root, "derivatives", "stats")
if not op.isdir(save_dir):
    os.mkdir(save_dir)
for epoch, name in zip((resp, aud, go), ("resp", "aud", "go")):
    epoch.mask = stats.time_perm_cluster(epoch.copy()._data, base.copy()._data,
                                         0.05, n_perm=1000, ignore_adjacency=1)
    resp_mask = mne.EvokedArray(resp.mask, resp.average().info)
    epoch.save(save_dir + f"/{subj}_{name}_mask-epo.fif", overwrite=True, fmt='double')
