"""
High Gamma Filter
===================================

Below is a code sample for extracting high gamma power from a raw data file
"""

from ieeg.navigate import channel_outlier_marker, trial_ieeg
from ieeg.io import raw_from_layout
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from bids import BIDSLayout
import mne

# %%
# Load Data
# ---------
bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
raw = raw_from_layout(layout,
                      subject="pt1",
                      preload=True,
                      extension=".vhdr")

# %%
# Preprocessing
# -------------

# Mark channel outliers as bad
channel_outlier_marker(raw, 5)

# Exclude bad channels
raw.drop_channels(raw.info['bads'])
good = raw.copy()
good.load_data()

# Remove intermediates for memory efficiency
del raw

# CAR (common average reference)
good.set_eeg_reference()

# %%
# High Gamma Filter
# -----------------
# Extract the epochs of interest

ev1 = trial_ieeg(good,
                 event="AD1-4, ATT1,2",
                 times=(-1, 2),
                 preload=True)
base = trial_ieeg(good,
                  event="onset",
                  times=(-1, 0.5),
                  preload=True)

# extract high gamma power
gamma.extract(ev1,
              copy=False,
              n_jobs=1)
gamma.extract(base,
              copy=False,
              n_jobs=1)

# trim 0.5 seconds on the beginning and end of the data (edge artifacts)
crop_pad(ev1, "500ms")
crop_pad(base, "500ms")

# %%
# Baseline Normalization
# ----------------------
# Z-score normalize `ev1` in-place to save memory

rescale(ev1, base,
        mode='zscore',
        copy=False)

# %%
# Plotting
# --------
resp_evoke = ev1.average()
resp_evoke.plot()
