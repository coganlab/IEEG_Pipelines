"""
Load IEEG Data Example
======================

This example demonstrates the discrepancy between the provided examples and the actual data loading process.

The provided examples:
1. Use `mne` library and `raw_from_layout` function to load IEEG data.
2. Use `mne` library and `read_raw` function to load IEEG data.

However, the actual data loading process involves using the `ieeg.io` module and the `get_data` and `raw_from_layout` functions.

"""

# %% Example 1
# ECoG data in BIDS file structure saved in the [BrainVision Core Data Format](https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/)
import mne
from ieeg.io import raw_from_layout
from bids import BIDSLayout
bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
raw1 = raw_from_layout(layout, subject="pt1", preload=True, extension=".vhdr")
raw1.plot()

# %% Example 2
# SEEG data saved in the [FIF file format](https://mne.tools/stable/auto_tutorials/io/plot_20_reading_eeg_data.html#sphx-glr-auto-tutorials-io-plot-20-reading-eeg-data-py)
import mne
misc_path = mne.datasets.misc.data_path()
raw2 = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')
raw2


# %% Actual Data Loading Process
# ECoG / SEEG data in BIDS file structure saved in the [European data format](https://www.edfplus.info/specs/edf.html)
import os
from ieeg.io import get_data, raw_from_layout

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

for subject in subjects:
    # do stuff with each subject data
    pass

print(subjects)

"""
The provided examples above do not reflect the actual data loading process used in this script.

To load the IEEG data, the following steps are taken:

1. Import the necessary modules:
    - `ieeg.io` module for data loading functions.
    - `os` module for file path manipulation.

2. Set the home directory and the root folder containing the BIDS formatted data.

3. Use the `get_data` function from `ieeg.io` to obtain a BIDSLayout object with the specified root folder and task name.

4. Get the list of subjects from the BIDSLayout object.

5. Iterate over each subject and perform the desired operations on the subject data.

Note: The actual code provided above is not executable as it contains placeholders and references to specific file paths on the user's computer.
"""
