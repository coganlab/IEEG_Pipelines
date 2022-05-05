# %%

import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import nilearn.plotting
from dipy.align import resample

import mne
from mne.datasets import fetch_fsaverage

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = op.join(sample_path, 'subjects')

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed

T1 = nib.load(op.join(misc_path, 'seeg', 'sample_seeg', 'mri', 'T1.mgz'))
viewer = T1.orthoview()
viewer.set_position(0, 9.9, 5.8)
viewer.figs[0].axes[0].annotate(
    'PC', (107, 108), xytext=(10, 75), color='white',
    horizontalalignment='center',
    arrowprops=dict(facecolor='white', lw=0.5, width=2, headwidth=5))
viewer.figs[0].axes[0].annotate(
    'AC', (137, 108), xytext=(246, 75), color='white',
    horizontalalignment='center',
    arrowprops=dict(facecolor='white', lw=0.5, width=2, headwidth=5))


def plot_overlay(image, compare, title, thresh=None):
    """Define a helper function for comparing plots."""
    image = nib.orientations.apply_orientation(
        np.asarray(image.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(image.affine))).astype(np.float32)
    compare = nib.orientations.apply_orientation(
        np.asarray(compare.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(compare.affine))).astype(np.float32)
    if thresh is not None:
        compare[compare < np.quantile(compare, thresh)] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.imshow(np.take(image, [image.shape[i] // 2], axis=i).squeeze().T,
                  cmap='gray')
        ax.imshow(np.take(compare, [compare.shape[i] // 2],
                          axis=i).squeeze().T, cmap='gist_heat', alpha=0.5)
        ax.invert_yaxis()
        ax.axis('off')
    fig.tight_layout()


CT_orig = nib.load(op.join(misc_path, 'seeg', 'sample_seeg_CT.mgz'))

# resample to T1's definition of world coordinates
CT_resampled = resample(moving=np.asarray(CT_orig.dataobj),
                        static=np.asarray(T1.dataobj),
                        moving_affine=CT_orig.affine,
                        static_affine=T1.affine)
plot_overlay(T1, CT_resampled, 'Unaligned CT Overlaid on T1', thresh=0.95)
del CT_resampled


reg_affine = np.array([
    [0.99270756, -0.03243313, 0.11610254, -133.094156],
    [0.04374389, 0.99439665, -0.09623816, -97.58320673],
    [-0.11233068, 0.10061512, 0.98856381, -84.45551601],
    [0., 0., 0., 1.]])
CT_aligned = mne.transforms.apply_volume_registration(CT_orig, T1, reg_affine)
plot_overlay(T1, CT_aligned, 'Aligned CT Overlaid on T1', thresh=0.95)
del CT_orig

CT_data = CT_aligned.get_fdata().copy()
CT_data[CT_data < np.quantile(CT_data, 0.95)] = np.nan
T1_data = np.asarray(T1.dataobj)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for ax in axes:
    ax.axis('off')
axes[0].imshow(T1_data[T1.shape[0] // 2], cmap='gray')
axes[0].set_title('MR')
axes[1].imshow(np.asarray(CT_aligned.dataobj)[CT_aligned.shape[0] // 2],
               cmap='gray')
axes[1].set_title('CT')
axes[2].imshow(T1_data[T1.shape[0] // 2], cmap='gray')
axes[2].imshow(CT_data[CT_aligned.shape[0] // 2], cmap='gist_heat', alpha=0.5)
for ax in (axes[0], axes[2]):
    ax.annotate('Subcutaneous fat', (110, 52), xytext=(100, 30),
                color='white', horizontalalignment='center',
                arrowprops=dict(facecolor='white'))
for ax in axes:
    ax.annotate('Skull (dark in MR, bright in CT)', (40, 175),
                xytext=(120, 246), horizontalalignment='center',
                color='white', arrowprops=dict(facecolor='white'))
axes[2].set_title('CT aligned to MR')
fig.tight_layout()
del CT_data, T1

subj_trans = mne.coreg.estimate_head_mri_t(
    'sample_seeg', op.join(misc_path, 'seeg'))

raw = mne.io.read_raw(op.join(misc_path, 'seeg', 'sample_seeg_ieeg.fif'))

gui = mne.gui.locate_ieeg(raw.info, subj_trans, CT_aligned,
                          subject='sample_seeg',
                          subjects_dir=op.join(misc_path, 'seeg'))
# The `raw` object is modified to contain the channel locations
# after closing the GUI and can now be saved
gui.close()  # close when done

# %%
# Let's do a quick sidebar and show what this looks like for ECoG as well.

T1_ecog = nib.load(op.join(misc_path, 'ecog', 'sample_ecog', 'mri', 'T1.mgz'))
CT_orig_ecog = nib.load(op.join(misc_path, 'ecog', 'sample_ecog_CT.mgz'))

# pre-computed affine from `mne.transforms.compute_volume_registration`
reg_affine = np.array([
    [0.99982382, -0.00414586, -0.01830679, 0.15413965],
    [0.00549597, 0.99721885, 0.07432601, -1.54316131],
    [0.01794773, -0.07441352, 0.99706595, -1.84162514],
    [0., 0., 0., 1.]])
# align CT
CT_aligned_ecog = mne.transforms.apply_volume_registration(
    CT_orig_ecog, T1_ecog, reg_affine)

raw_ecog = mne.io.read_raw(op.join(misc_path, 'ecog', 'sample_ecog_ieeg.fif'))
# use estimated `trans` which was used when the locations were found previously
subj_trans_ecog = mne.coreg.estimate_head_mri_t(
    'sample_ecog', op.join(misc_path, 'ecog'))
gui = mne.gui.locate_ieeg(raw_ecog.info, subj_trans_ecog, CT_aligned_ecog,
                          subject='sample_ecog',
                          subjects_dir=op.join(misc_path, 'ecog'))
