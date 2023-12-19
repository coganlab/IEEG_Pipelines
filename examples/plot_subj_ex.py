"""
Subject Brain Plotting Example
===============================

Below is a code sample for plotting brain plots
"""
import os.path
import mne
from ieeg.viz.mri import plot_subj

misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')
subjects_dir = os.path.join(sample_path, 'subjects')

plot_subj(raw, subjects_dir, labels_every=None)
