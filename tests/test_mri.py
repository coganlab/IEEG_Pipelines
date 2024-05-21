import os
import mne
import vtk
import pytest
from ieeg.viz.mri import plot_subj, plot_on_average

misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"
os.environ["MNE_3D_OPTION_ANTIALIAS"] = "false"

if not os.path.exists(os.path.join(subjects_dir, "1")):
    os.symlink(os.path.join(subjects_dir, 'sample'),
               os.path.join(subjects_dir, '1'))

raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")


@pytest.mark.parametrize(
    'args, expected', [
        ({}, None),
        ({'labels_every': 1}, None),
        ({'hemi': 'split'}, None),
        ({'hemi': 'lh'}, None),
        ({'size': 0.2}, None),
        ({'background': 'red'}, None),
        ({'transparency': 0.2}, None),
        ({'color': (1, 0, 0)}, None),
        ({'picks': ['LSTG 1', 'LSTG 2', 'LSTG 3', 'LSTG 4', 'RAHP 1',
                    'RAHP 2', 'RAHP 3', 'RAHP 4', 'RAHP 5', 'RAHP 6']}, None),
        ({'picks': ['LSTG 1', 'LSTG 2', 'LSTG 3'], 'hemi': 'rh'}, None),
        ({'picks': [0, 1, 2, 3, 4, 5, 6, 7, 8]}, None),
        ({'picks': [0, 1, 2, 3, 4]}, None),
        ({'hemi': 'rh'}, None),
        # Fails bc no wm parcellation file
        # ({'no_wm': True}, None),
    ]
)
def test_plot_seeg_inst(args, expected):
    plot_subj(inst=raw, subj_dir=subjects_dir, show=False, **args)


@pytest.mark.parametrize(
    'args, expected', [
        ({}, None),
        ({'label_every': 1}, None),
        ({'hemi': 'both'}, None),
        ({'hemi': 'split'}, None),
        ({'hemi': 'lh'}, None),
        ({'size': 0.2}, None),
        ({'background': 'red'}, None),
        ({'transparency': 0.2}, None),
        ({'color': (1, 0, 0)}, None),
        ({'picks': ['LSTG 1', 'LSTG 2', 'LSTG 3', 'LSTG 4', 'RAHP 1',
                    'RAHP 2', 'RAHP 3', 'RAHP 4', 'RAHP 5', 'RAHP 6']}, None),
        ({'picks': ['LSTG 1', 'LSTG 2', 'LSTG 3'], 'hemi': 'rh'}, None),
        ({'picks': [0, 1, 2, 3, 4, 5, 6, 7, 8]}, None),
        ({'picks': [0, 1, 2, 3, 4]}, None),
        ({'hemi': 'rh'}, None),
    ]
)
def test_plot_avg(args, expected):
    plot_on_average(sigs=raw, subj_dir=subjects_dir, show=False, **args)


# if on windows, finalize closing the vtk window
if os.name == 'nt':
    vtk.vtkRenderWindow().Finalize()
    vtk.vtkObject.GlobalWarningDisplayOff()
