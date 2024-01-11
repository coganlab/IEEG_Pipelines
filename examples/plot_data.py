"""
Load IEEG Data Example
======================

 This example demonstrates the discrepancy between the provided examples and the
 actual data loading process.

The provided examples:
    1. Use `raw_from_layout` function to load ECoG data.
    2. Use `mne.io.read_raw` function to load SEEG data.
    3. Use `get_data` and `raw_from_layout` functions to load ECoG or SEEG data

.. note::
 Only example 3 is representative of the actual data loading process for
 CoganLab data. The other two examples are provided for comparison purposes, as
 well as because CoganLab data cannot be loaded in these docs. Going forward,
 we will be using examples 1 and 2 to demonstrate the data loading process,
 while you should use example 3 to load CoganLab data.

"""

# %%
# Example 1 (BrainVision)
# -----------------------
# ECoG data in BIDS file structure saved in the `BrainVision Core Data Format
# <https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/>`_
import mne
from ieeg.io import raw_from_layout
from bids import BIDSLayout
bids_root = mne.datasets.epilepsy_ecog.data_path()
layout = BIDSLayout(bids_root)
raw1 = raw_from_layout(layout,
                       subject="pt1",
                       preload=True,
                       extension=".vhdr")
raw1.plot()

# %%
# Example 2 (FIF)
# ---------------
# SEEG data saved in the `FIF file format
# <https://mne.tools/stable/auto_tutorials/io/plot_20_reading_eeg_data.html#sphx-glr-auto-tutorials-io-plot-20-reading-eeg-data-py>`_
import mne
misc_path = mne.datasets.misc.data_path()
raw2 = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')
raw2

# %%
# Example 3 (EDF)
# ---------------
# ECoG / SEEG data in BIDS file structure saved in the `European data format
# <https://www.edfplus.info/specs/edf.html>`_
import os
from ieeg.io import get_data, raw_from_layout

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("SentenceRep", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

for subject in subjects:
    raw = raw_from_layout(layout, subject=subject, preload=True)
    # do stuff with each subject data

print(subjects)

# %%
# Examples 1 and 2 above do not reflect the data loading process
# for CoganLab data.
#
# To load the IEEG data, the following steps are taken:
#
#     1. Import the necessary modules:
#         - `ieeg.io` module for data loading functions.
#         - `os` module for file path manipulation.
#     2. Set the home directory and the root folder containing the BIDS formatted data.
#     3. Use the `get_data` function to obtain a BIDSLayout object with the specified root folder and task name.
#     4. Get the list of subjects from the BIDSLayout object.
#     5. Iterate over each subject and perform the desired operations on the subject data.
#
# .. note:: The actual code provided above is not executable as it
#   contains placeholders and references to specific file paths
#   on the user's computer.
