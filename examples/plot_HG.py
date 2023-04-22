"""
Example of High Gamma Filter
===================================

Below is a code sample for extracting high gamma power from a raw data file
"""

from ieeg.navigate import channel_outlier_marker, trial_ieeg
from ieeg.io import raw_from_layout
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from bids import BIDSLayout
import mne

# %% Load Data
bids_root = mne.datasets.epilepsy_ecog.data_path()
# sample_path = mne.datasets.sample.data_path()
layout = BIDSLayout(bids_root)
raw = raw_from_layout(layout, subject="pt1", preload=True,
                      extension=".vhdr")

# %% Some preprocessing

# Mark channel outliers as bad
channel_outlier_marker(raw, 5)

# Exclude bad channels
raw.drop_channels(raw.info['bads'])
good = raw.copy()
good.load_data()

# Remove intermediates from mem
del raw

# CAR
good.set_eeg_reference()

# %% High Gamma Filter

ev1 = trial_ieeg(good, "AD1-4, ATT1,2", (-1, 2), "onset", (-1, 0.5))

HG_ev1 = gamma.extract(ev1)
crop_pad(HG_ev1, "500ms")

# %% plotting
resp_evoke = HG_ev1.average()
resp_evoke.plot()
