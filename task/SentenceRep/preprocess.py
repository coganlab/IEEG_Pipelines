import ieeg.viz.utils
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import crop_empty_data, channel_outlier_marker, trial_ieeg
from ieeg.timefreq import gamma, utils
from ieeg.calc import stats, scaling
import numpy as np
import os.path as op
import os
import mne

# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 18

# %% Load the data
TASK = "SentenceRep"
subj = "D" + str(subject).zfill(4)
layout = get_data("SentenceRep", root=LAB_root)
filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                       extension='.edf', desc='clean', preload=False)

# %% Crop raw data to minimize processing time
new = crop_empty_data(filt, )

# Mark channel outliers as bad
new.info['bads'] = channel_outlier_marker(new, 3, 2)

# Exclude bad channels
good = new.copy().drop_channels(new.info['bads'])
good.load_data()

# CAR
ch_type = filt.get_channel_types(only_data_chs=True)[0]
good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

# Remove intermediates from mem
del new

# %% fix SentenceRep events
from events import fix_annotations  # noqa E402
fix_annotations(good)

# %% High Gamma Filter and epoching
out = []
for epoch, t in zip(("Start", "Word/Response", "Word/Audio/LS",
                     "Word/Audio/LM", "Word/Audio/JL", "Word/Speak",
                     "Word/Mime", "Word/Audio/JL"),
                    ((-0.5, 0), (-1, 1), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5),
                     (-0.5, 1.5), (-0.5, 1.5), (1, 3))):
    times = [None, None]
    times[0] = t[0] - 0.5
    times[1] = t[1] + 0.5
    trials = trial_ieeg(good, epoch, times, preload=True, outliers=10)
    gamma.extract(trials, copy=False, n_jobs=1)
    utils.crop_pad(trials, "0.5s")
    trials.resample(100)
    trials.filenames = good.filenames
    out.append(trials)
    # if len(out) == 2:
    #     break

base = out.pop(0)

# %% run time cluster stats

save_dir = op.join(layout.root, "derivatives", "stats")
if not op.isdir(save_dir):
    os.mkdir(save_dir)
mask = dict()
for epoch, name in zip(out, ("resp", "aud_ls", "aud_lm", "aud_jl", "go_ls",
                             "go_lm", "go_jl")):
    sig1 = epoch.get_data()
    sig2 = base.get_data()
    sig2 = np.pad(sig2, ((0, 0), (0, 0), (0, sig1.shape[-1] - sig2.shape[-1])),
                  mode='reflect')
    mask[name] = stats.time_perm_cluster(sig1, sig2, 0.05,
                                         n_perm=1000, ignore_adjacency=1)
    epoch_mask = mne.EvokedArray(mask[name], epoch.average().info)
    power = scaling.rescale(epoch, base, copy=True)
    power.save(save_dir + f"/{subj}_{name}_power-epo.fif", overwrite=True,
               fmt='double')
    z_score = scaling.rescale(epoch, base, 'zscore', copy=True)
    z_score.save(save_dir + f"/{subj}_{name}_zscore-epo.fif", overwrite=True,
                 fmt='double')
    epoch_mask.save(save_dir + f"/{subj}_{name}_mask-ave.fif", overwrite=True)

# %% Plot
import matplotlib.pyplot as plt  # noqa E402
plt.imshow(mask['go_ls'])
