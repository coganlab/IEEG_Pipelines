"""
Subject Brain Plotting Example
===============================

Below is a code sample for plotting brain plots
"""
import os.path
import mne
from ieeg.viz.mri import plot_subj, plot_on_average

misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')
subjects_dir = os.path.join(sample_path, 'subjects')

# %%
# make matching subject directory for subject if it doesn't exist
if not os.path.exists(os.path.join(subjects_dir, "1")):
    os.symlink(os.path.join(subjects_dir, 'sample'),
               os.path.join(subjects_dir, '1'))

# %%
# Load the Data
# -------------

raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")

# %%
# Plot the Brain
# --------------
fig1 = plot_subj(raw, subj_dir=subjects_dir)
mne.viz.set_3d_view(fig1, azimuth=150, elevation=70, focalpoint="auto",
                    distance="auto")

# %%
# Plot the data on an average brain
# ---------------------------------

fig2 = plot_on_average(raw, subj_dir=subjects_dir, rm_wm=False, hemi='both')
mne.viz.set_3d_view(fig2, azimuth=150, elevation=70, focalpoint="auto",
                    distance="auto")
