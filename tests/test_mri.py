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


def test_plot_seeg():
    from ieeg.viz.mri import plot_subj
    plot_subj(raw, subj_dir=subjects_dir, show=False)


def test_plot_avg():
    from ieeg.viz.mri import plot_on_average
    plot_on_average(raw, subj_dir=subjects_dir, show=False, rm_wm=False,
                    hemi='both')


# if on windows, finalize closing the vtk window
if os.name == 'nt':
    vtk.vtkRenderWindow().Finalize()
    vtk.vtkObject.GlobalWarningDisplayOff()
