"""
PCA-LDA Decoding
===================================

Takes high gamma filtered data with event labels from multiple subjects and
performs joint pca decoding
"""

from ieeg.decoding.models import PcaLdaClassification
from ieeg.arrays.label import LabeledArray
from sklearn.metrics import ConfusionMatrixDisplay
from ieeg.decoding.decode import Decoder, classes_from_labels
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

cats, labels = classes_from_labels(arr.labels[0])
decoder = Decoder(cats, explained_variance=0.80,
                  oversample=True, n_splits=5, n_repeats=100)
cm = decoder.cv_cm(arr.__array__().swapaxes(0, 1), labels, normalize='true')
cm = np.mean(cm, axis=0)

# %% Plot the Confusion Matrix

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cats.keys())
disp.plot(ax=ax)
