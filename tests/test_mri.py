import os

import mne
import vtk

misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"
os.environ["MNE_3D_OPTION_ANTIALIAS"] = "false"

if not os.path.exists(os.path.join(subjects_dir, "1")):
    os.symlink(os.path.join(subjects_dir, 'sample'),
               os.path.join(subjects_dir, '1'))
raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")

import pytest

@pytest.mark.parametrize(
    'args, expected', [
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'labels_every': 1}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'hemi': 'split'}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'hemi': 'lh'}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'size': 0.2}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'background': 'red'}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'transparency': 0.2}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'color': (1, 0, 0)}, None),
    # Generates a plot when show=True but yields error
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'picks':[0,1,2,3,4]}, None),
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'hemi': 'rh'}, None),
    # Fails ompletely
    ({'inst': raw, 'subj_dir': subjects_dir, 'show': False, 'no_wm': True}, None),
])
def test_plot_seeg_inst(args, expected):
    from ieeg.viz.mri import plot_subj
    plot_subj(**args)

@pytest.mark.parametrize(
    'args, expected', [
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'label_every': 1}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'hemi': 'both'}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'hemi': 'split'}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'hemi': 'lh'}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'size': 0.2}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'background': 'red'}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'transparency': 0.2}, None),
    ({'sigs': raw, 'subj_dir': subjects_dir, 'show': False, 'color': (1, 0, 0)}, None),
])
def test_plot_avg(args, expected):
    from ieeg.viz.mri import plot_on_average
    plot_on_average(**args)


# if on windows, finalize closing the vtk window
if os.name == 'nt':
    vtk.vtkRenderWindow().Finalize()
    vtk.vtkObject.GlobalWarningDisplayOff()
