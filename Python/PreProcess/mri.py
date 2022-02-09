import os.path as op

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from bids import BIDSLayout
from dipy.align import resample

from utils import LAB_root, PathLike


def plot_overlay(image: nib.Nifti1Image, compare: nib.Nifti1Image,
                 title: str, thresh: float = None):
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


def allign_mri(t1_path: PathLike, ct_path: PathLike, my_raw: mne.io.Raw,
               sub_id: str, subj_dir: PathLike = None):
    if subj_dir is None:
        subj_dir = op.join(LAB_root, "ECoG_Recon_Full")
    T1 = nib.load(t1_path)
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
    CT_orig = nib.load(ct_path)
    CT_resampled = resample(moving=np.asarray(CT_orig.dataobj),
                            static=np.asarray(T1.dataobj),
                            moving_affine=CT_orig.affine,
                            static_affine=T1.affine)
    plot_overlay(T1, CT_resampled, 'Unaligned CT Overlaid on T1', thresh=0.95)
    del CT_resampled
    reg_affine, _ = mne.transforms.compute_volume_registration(
        CT_orig, T1, pipeline='rigids')
    CT_aligned = mne.transforms.apply_volume_registration(CT_orig, T1,
                                                          reg_affine)
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
    axes[2].imshow(CT_data[CT_aligned.shape[0] // 2], cmap='gist_heat',
                   alpha=0.5)
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
    subj_trans = mne.coreg.estimate_head_mri_t(sub_id, subjects_dir=subj_dir)
    gui = mne.gui.locate_ieeg(my_raw.info, subj_trans, CT_aligned,
                              subject=sub_id, subjects_dir=subj_dir)
    return gui


if __name__ == "__main__":
    BIDS_root = op.join(LAB_root, "BIDS-1.3_Phoneme_sequencing", "BIDS")
    layout = BIDSLayout(BIDS_root)
    T1_path = layout.get(return_type="path", subject="D0022", type="T1w")[0]
    CT_path = layout.get(return_type="path", subject="D0022", type="CT")[0]
    filt = mne.io.read_raw_fif("D22_filt_ieeg.fif")
    allign_mri(T1_path, CT_path, filt, "D0022")
