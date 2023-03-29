import numpy as np

from PreProcess.navigate import get_data, crop_data, \
    channel_outlier_marker, raw_from_layout, trial_ieeg
import os.path as op
import matplotlib as mpl
try:
    mpl.use("TkAgg")
except ImportError:
    pass

# %% Load the data
TASK = "SentenceRep"
sub_num = 29
subj = "D" + str(sub_num).zfill(4)
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
from task.SentenceRep.events import fix_annotations
fix_annotations(good)

# %% High Gamma Filter and epoching
out = []
from PreProcess.timefreq import gamma, utils, fastmath
for epoch, t in zip(("Start", "Word/Response", "Word/Audio", "Word/Speak"),
                    ((-0.5, 0), (-1, 1), (-0.5, 1.5), (-0.5, 1.5))):
    times = [None, None]
    times[0] = t[0] - 0.5
    times[1] = t[1] + 0.5
    trials = trial_ieeg(good, epoch, times, preload=True)
    gamma.extract(trials, copy=False)
    utils.crop_pad(trials, "0.5s")
    out.append(trials)
    if len(out) == 2:
        break
# resp = fastmath.rescale(out[1], out[0], copy=True)
# aud = fastmath.rescale(out[2], out[0], copy=True)
# go = fastmath.rescale(out[3], out[0], copy=True)
resp = out[1].copy()
resp.decimate(20)
base = out[0].copy()
base.decimate(20)
z_vals = fastmath.rescale(resp, base, 'zscore', True)
z = z_vals.average()
power = fastmath.rescale(resp, base, 'mean', True).average()
# %%
def time_perm_cluster(sig1: np.ndarray, sig2: np.ndarray, n_perm: int = 1000,
                      tails: int = 1, axis: int = 0) -> np.ndarray:

    # Concatenate the two signals for trial shuffling
    all_trial = np.concatenate((sig1, sig2), axis=axis)
    labels = np.concatenate((np.zeros(sig1.shape[axis]),
                             np.ones(sig2.shape[axis])))

    all_labels = np.array([np.random.permutation(labels) == 1 for _ in range(n_perm)])

    # Calculate the observed difference
    obs_diff = np.mean(sig1, axis) - np.mean(sig2, axis)

    # Calculate the difference between the two groups averaged across
    # trials at each time point
    fake_sig1 = np.array([np.take(all_trial, np.where(
        all_labels[i] == 0)[0], axis=axis) for i in range(n_perm)])
    fake_sig2 = np.array([np.take(all_trial, np.where(
        all_labels[i] == 1)[0], axis=axis) for i in range(n_perm)])
    diff = np.mean(fake_sig1, axis=axis+1) - np.mean(fake_sig2, axis=axis+1)
    if tails == 1:
        larger = diff > obs_diff
    elif tails == 2:
        larger = np.abs(diff) > np.abs(obs_diff)
    else:
        raise ValueError('tails must be 1 or 2')

    # Calculate the p-value
    p = np.sum(larger, axis=0) / n_perm

    return p
# %%
sigA = resp.copy()._data

# sigB = fastmath.make_data_same(base._data.copy(), sigA, ignore_axis=0)

# sigB = np.array([np.repeat(np.array([base._data.copy()[:,i,:].flatten(
#     )]).T,sigA.shape[2], axis=1) for i in range(sigA.shape[1])])

trials = int(base._data.copy()[:, 0, :].size / sigA.shape[-1])
sigB = np.full((trials, *sigA.shape[1:]), np.nan)
for i in range(sigA.shape[1]):
    sigB[:, i, :].flat = base._data.copy()[:, i, :].flat
del trials

# p_vals = time_perm_cluster(sigA, sigB, 10000)

# %%
import mne
func = lambda s1, s2: np.mean(s1, axis=0) - np.mean(s2, axis=0)
# find the 95th percentile of the null distribution
threshold = np.percentile(func(sigA, sigB), 95)
Z, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
    [sigA, sigB], out_type='mask', n_jobs =-1, stat_fun=func,
    n_permutations=1000, threshold=threshold, tail=1, t_power=0)
clust = np.sum(np.array(clusters), 0)
mpl.pyplot.imshow(clust)

# %%
# Z = np.full(sigA.shape[1:], 0.)
# clust = []
# for i in range(threshold.shape[0]):
#     Z[i], clusters, cluster_p_values, H0 = \
#         mne.stats.permutation_cluster_test([sigA[:,i,:], sigB[:,i,:]], out_type='mask', n_jobs =-1, stat_fun=func,
#                                  n_permutations=1000, threshold=threshold[i], tail=1
#                                            )
#     if not clusters:
#         clust.append(np.zeros((sigA.shape[2],)))
#     else:
#         clust.append(np.array(clusters))
# # clust = np.array(clust)
# clust = np.sum(np.array(clust),0)
# mpl.pyplot.imshow(Z)
