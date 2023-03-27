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
    trials_perm = np.array([all_trial for _ in range(n_perm)])

    # Calculate the observed difference
    obs_diff = np.mean(sig1, axis) - np.mean(sig2, axis)

    # Calculate the difference between the two groups averaged across
    # trials at each time point
    fake_sig1 = np.array([np.take(all_trial, np.where(
        all_labels[i] == 0)[0], axis=axis) for i in range(n_perm)])
    fake_sig2 = np.array([np.take(all_trial, np.where(
        all_labels[i] == 1)[0], axis=axis) for i in range(n_perm)])
    diff = np.mean(fake_sig1, axis=axis) - np.mean(fake_sig2, axis=axis)
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
import mne
import scipy
# resp_evoke = resp.average()
# resp_evoke.plot()
sigA = resp.copy()._data
# sigB = fastmath.make_data_same(base._data.copy(), sigA, ignore_axis=0)
sigB = np.array([np.repeat(np.array([base._data.copy()[:,i,:].flatten()]
                                    ).T,sigA.shape[2], axis=1) for i in range(sigA.shape[1])])
sigB = np.swapaxes(sigB, 0, 1)
# sigA = np.swapaxes(sigA, 1, 2)
# sigB = np.swapaxes(sigB, 1, 2)
# sigC = np.swapaxes(power._data, 1, 2)
# allsig = np.concatenate([sigA[:, :, 0], sigB[:, :, 0]], axis=0)
p_vals = time_perm_cluster(sigA, sigB, 100)
# sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(power.info, None)
# df = 150-1  # degrees of freedom
# func = lambda x : scipy.stats.kruskal(*x)
# t_lim = scipy.stats.distributions.t.ppf(1 - 0.001 / 2, df=df)
# F_obs, clusters, cluster_p_values, H0 = \
#     mne.stats.permutation_cluster_test([sigA, sigB], out_type='mask', n_jobs =-1, stat_fun=scipy.stats.kruskal,
#                              n_permutations=100, threshold=10, tail=1, adjacency=sensor_adjacency
#                                        )
# # F_obs, clusters, cluster_p_values, H0 = \
# #     mne.stats.permutation_cluster_1samp_test([z_vals._data.copy()], out_type='mask', n_jobs =-1, # stat_fun=shuffle_test,
# #                              n_permutations=100, threshold=None, tail=1, adjacency=None)
# clust = np.sum(np.array(clusters),0)
# mpl.pyplot.imshow(clust)