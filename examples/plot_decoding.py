"""
PCA-LDA Decoding
===================================

Takes high gamma filtered data with event labels from multiple subjects and
performs joint pca decoding
"""

from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
from ieeg.navigate import channel_outlier_marker, trial_ieeg, outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.mt_filter import line_filter
import mne
import numpy as np
import matplotlib.pyplot as plt


# %%
# Load Data
# ---------
misc_path = mne.datasets.misc.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif',
                      preload=True)

# %%
# Filter the data to remove line noise
# ------------------------------------

line_filter(raw, mt_bandwidth=10., n_jobs=1, copy=False, verbose=10,
            filter_length='700ms', freqs=[60], notch_widths=20)
# line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
#             filter_length='7s', freqs=[60, 120, 180, 240],
#             notch_widths=20)
raw.plot()

# %%
# Preprocessing
# -------------

# Mark channel outliers as bad
channel_outlier_marker(raw, 3, 2)

# Exclude bad channels, then load the good channels into memory
raw.drop_channels(raw.info['bads'])
good = raw.copy()
good.load_data()

# CAR (common average reference)
good.set_eeg_reference()
del raw

# %%
# High Gamma Filter
# -----------------

# extract the epochs of interest
ev1 = trial_ieeg(good, ["Fixation", "ISI Onset", "Go Cue", "Response"],
                 (-0.6, 0.7), preload=True)
base = trial_ieeg(good, "Fixation", (-1, 0.5), preload=True)

outliers_to_nan(ev1, outliers=10)
outliers_to_nan(base, outliers=10)

# extract high gamma power
gamma.extract(ev1, passband=(70, 150), copy=False, n_jobs=1)
gamma.extract(base, passband=(70, 150), copy=False, n_jobs=1)

# trim 0.5 seconds on the beginning and end of the data (edge artifacts)
crop_pad(ev1, "500ms")
crop_pad(base, "500ms")

# ev1["Response"]._data += np.sin(np.linspace(0, 2*np.pi, 401)) * 10
# %%
# reformat data for decoding
# --------------------------

rescale(ev1, base,
        mode='zscore',
        copy=False)

# create a LabeledArray
arr = LabeledArray.from_signal(ev1)

# %%
# Create a Decoder
# ----------------


class Decoder(PcaLdaClassification, MinimumNaNSplit):

    def __init__(self, categories: dict, *args,
                 n_splits: int = 5,
                 n_repeats: int = 10,
                 oversample: bool = True,
                 max_features: int = float("inf"),
                 **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats)
        if not oversample:
            self.oversample = lambda x, func, axis: x
        self.categories = categories
        self.max_features = max_features

    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(
                0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            for i in set(labels):
                # fill in train data nans with random combinations of
                # existing train data trials (mixup)
                idx[obs_axs] = y_train == i
                x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)],
                                                      axis=obs_axs,
                                                      func=mixup)

            # fill in test data nans with noise from distribution
            is_nan = np.isnan(x_test)
            x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))

            # feature selection
            train_in = flatten_features(x_train, obs_axs)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features:
                tidx = np.random.choice(train_in.shape[1], self.max_features,
                                        replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]

            # fit model and score results
            self.fit(train_in, y_train)
            pred = self.predict(test_in)
            rep, fold = divmod(f, self.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)

        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1)
        if normalize == 'true':
            divisor = np.sum(matk, axis=-1, keepdims=True)
        elif normalize == 'pred':
            divisor = np.sum(matk, axis=-2, keepdims=True)
        elif normalize == 'all':
            divisor = self.n_repeats
        else:
            divisor = 1
        return matk / divisor


def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.copy()
    return out.reshape(out.shape[0], -1)


def classes_from_labels(labels: np.ndarray, delim: str = '-', which: int = 0,
                        crop: slice = slice(None)) -> tuple[dict, np.ndarray]:
    class_ids = np.array([k.split(delim)[which][crop] for k in labels])
    classes = {k: i for i, k in enumerate(np.unique(class_ids))}
    return classes, np.array([classes[k] for k in class_ids])


cats, labels = classes_from_labels(arr.labels[0])
decoder = Decoder(cats, 0.80, oversample=True, n_splits=5, n_repeats=100)
cm = decoder.cv_cm(arr.__array__().swapaxes(0, 1), labels, normalize='true')
cm = np.mean(cm, axis=0)

# %% Plot the Confusion Matrix

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cats.keys())
disp.plot(ax=ax)
