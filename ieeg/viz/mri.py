import os.path as op
from typing import Union, List

import matplotlib
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np

from ieeg import Signal, PathLike


def plot_overlay(image: nib.Nifti1Image, compare: nib.Nifti1Image,
                 title: str, thresh: float = None):
    """Plots an overlay of two images

    Parameters
    ----------
    image : nib.Nifti1Image
        The image to plot
    compare : nib.Nifti1Image
        The image to overlay
    title : str
        The title of the plot
    thresh : float, optional
        The threshold to apply to the overlay, by default None
    """
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


def allign_CT(t1_path: PathLike, ct_path: PathLike, reg_affine=None
              ) -> nib.spatialimages.SpatialImage:
    """Alligns a CT scan to a T1 scan

    Parameters
    ----------
    t1_path : PathLike
        The path to the T1 scan
    ct_path : PathLike
        The path to the CT scan
    reg_affine : np.ndarray, optional
        The affine to use for registration, by default None

    Returns
    -------
    nib.spatialimages.SpatialImage
        The alligned CT scan
    """
    T1 = nib.load(t1_path)
    CT_orig = nib.load(ct_path)
    sdr_morph = None
    if reg_affine is None:
        reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
            CT_orig, T1, pipeline='all')
    CT_aligned = mne.transforms.apply_volume_registration(CT_orig, T1,
                                                          reg_affine,
                                                          sdr_morph)
    return CT_aligned


def show_brain(my_raw: Signal, trans: mne.transforms.Transform,
               sub_id: PathLike, subjects_dir: PathLike = None,
               overwrite: bool = False):
    """Shows the brain with the electrodes projected onto it

    Parameters
    ----------
    my_raw : Signal
        The data to plot
    trans : mne.transforms.Transform
        The transformation to use
    sub_id : PathLike
        The subject id
    subjects_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'
    overwrite : bool, optional
        Whether to overwrite the watershed bem, by default False
    """
    subjects_dir = get_sub_dir(subjects_dir)
    try:
        mne.bem.make_watershed_bem(sub_id, subjects_dir, overwrite=overwrite)
    except RuntimeError:
        pass
    my_raw.info = mne.preprocessing.ieeg.project_sensors_onto_brain(
        my_raw.info, trans, sub_id, subjects_dir=subjects_dir)
    brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white')
    brain = mne.viz.Brain(sub_id, surf='white', subjects_dir=subjects_dir,
                          title='Projection', **brain_kwargs)
    brain.add_sensors(my_raw.info, trans=trans)
    view_kwargs = dict(azimuth=60, elevation=100, distance=350,
                       focalpoint=(0, 0, -15))
    brain.show_view(**view_kwargs)


def head_to_mni(inst: Signal, sub: str, subj_dir: PathLike = None):
    """Transforms the head coordinates to MNI Talairach coordinates

    Parameters
    ----------
    inst : Union[mne.io.Raw, mne.Epochs, mne.Evoked]
        The data to transform
    sub : str
        The subject id
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'
    """
    subj_dir = get_sub_dir(subj_dir)
    montage = inst.get_montage()

    # first we need a head to mri transform since the data is stored in "head"
    # coordinates, let's load the mri to head transform and invert it
    head_mri_t = mne.coreg.estimate_head_mri_t(sub, subj_dir)
    # apply the transform to our montage
    montage.apply_trans(head_mri_t)

    montage.add_estimated_fiducials(sub, subj_dir)
    # now let's load our Talairach transform and apply it
    mri_mni_t = mne.read_talxfm(sub, subj_dir)
    montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)

    # for fsaverage, "mri" and "mni_tal" are equivalent and, since
    # we want to plot in fsaverage "mri" space, we need use an identity
    # transform to equate these coordinate frames
    montage.apply_trans(
        mne.transforms.Transform(fro='mni_tal', to='mri', trans=np.eye(4)))
    # montage.add_mni_fiducials(sub, subj_dir)
    inst.set_montage(montage)


def get_sub_dir(subj_dir: PathLike = None):
    """Gets the subjects directory

    Parameters
    ----------
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'

    Returns
    -------
    PathLike
        The subjects directory
    """
    if subj_dir is None:
        from os import path
        HOME = path.expanduser("~")
        LAB_root = path.join(HOME, "Box", "CoganLab")
        subj_dir = op.join(LAB_root, "ECoG_Recon_Full")
    return subj_dir


def plot_gamma(evoked: mne.Evoked, subjects_dir: PathLike = None, **kwargs):
    """Plots the gamma power over time

    Parameters
    ----------
    evoked : mne.Evoked
        The data to plot
    subjects_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon_Full'
    **kwargs
        Additional arguments to pass to plot_on_average
        """
    data = evoked.copy().filter(30, 150).apply_hilbert(envelope=True)._data
    fig = plot_on_average(evoked.info, subjects_dir=subjects_dir, **kwargs)
    mne.viz.set_3d_view(fig, azimuth=0, elevation=70)

    xy, im = mne.viz.snapshot_brain_montage(fig, evoked.info)
    # convert from a dictionary to array to plot
    xy_pts = np.vstack([xy[ch] for ch in evoked.info['ch_names']])

    # get a colormap to color nearby points similar colors
    cmap = matplotlib.cm.get_cmap('viridis')

    # create the figure of the brain with the electrode positions
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Gamma power over time', size='large')
    ax.imshow(im)
    ax.set_axis_off()

    # normalize gamma power for plotting
    gamma_power = -100 * data / data.max()
    # add the time course overlaid on the positions
    x_line = np.linspace(-0.025 * im.shape[0], 0.025 * im.shape[0],
                         data.shape[1])
    for i, pos in enumerate(xy_pts):
        x, y = pos
        color = cmap(i / xy_pts.shape[0])
        ax.plot(x_line + x, gamma_power[i] + y, linewidth=0.5, color=color)


def plot_on_average(sigs: dict[str, Signal], subj_dir: PathLike = None,
                    surface: List[str] = None) -> matplotlib.figure.Figure:

    subj_dir = get_sub_dir(subj_dir)
    if surface is None:
        surface = 'pial'
    brain = mne.viz.Brain('fsaverage', subjects_dir=subj_dir,
                          background='grey', surf=surface)
    for subj, inst in sigs.items():
        new = inst.copy()
        montage = new.get_montage()

        # estimate head->mri transform
        subj_trans = mne.coreg.estimate_head_mri_t(op.join(
            get_sub_dir(), subj))
        montage.apply_trans(subj_trans)
        subject_brain = nib.load(
            op.join(get_sub_dir(), subj, 'mri', 'brain.mgz'))
        template_brain = nib.load(
            op.join(get_sub_dir(), 'fsaverage', 'mri', 'brain.mgz'))
        zooms = dict(translation=10, rigid=10, affine=10, sdr=5)
        reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
            subject_brain, template_brain, zooms=zooms, verbose=True)

        # warp the montage
        montage_warped = mne.preprocessing.ieeg.warp_montage(
            montage, subject_brain, template_brain, reg_affine, sdr_morph)

        # first we need to add fiducials so that we can define the "head" coordinate
        # frame in terms of them (with the origin at the center between LPA and RPA)
        montage_warped.add_estimated_fiducials('fsaverage', subj_dir)

        # compute the head<->mri ``trans`` now using the fiducials
        trans = mne.channels.compute_native_head_t(montage_warped)
        new.set_montage(montage)
        brain.add_sensors(new.info, trans)

    return brain


def plot_subj(inst: Signal, sub: str, subj_dir: PathLike = None,
              picks: list[str | int] = None):

    subj_dir = get_sub_dir(subj_dir)
    new = inst.copy().pick(picks)
    montage = new.get_montage()
    montage.apply_trans(mne.transforms.Transform(fro='unknown', to='head'))
    new.set_montage(montage)
    trans = mne.transforms.Transform(fro='head', to='mri')
    mne.viz.plot_alignment(new.info, subject=sub, trans=trans,
                           subjects_dir=subj_dir, surfaces='brain',
                           coord_frame='head')


def gen_labels(inst: Signal, sub: str, subj_dir: PathLike = None,
               picks: list[str | int] = None) -> dict[str, list]:

    subj_dir = get_sub_dir(subj_dir)
    montage = inst.get_montage()
    montage.apply_trans(mne.transforms.Transform(fro='ras', to='mri'))
    aseg = 'aparc+aseg'  # parcellation/anatomical segmentation atlas
    labels, colors = mne.get_montage_volume_labels(
        montage, sub, subjects_dir=subj_dir, aseg=aseg)

    if picks is not None:
        for i, key in enumerate(labels.keys()):
            if not any((i, key) in picks):
                labels.pop(key)

    return labels


if __name__ == "__main__":
    from ieeg.io import get_data, raw_from_layout
    from os import path
    HOME = path.expanduser("~")
    LAB_root = path.join(HOME, "Box", "CoganLab")
    # %% Set up logging
    log_filename = "output.log"
    # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    TASK = "SentenceRep"
    sub_num = 29
    layout = get_data(TASK, root=LAB_root)
    subj_dir = op.join(LAB_root, "ECoG_Recon_Full")
    sub_pad = "D00{}".format(sub_num)
    sub = "D{}".format(sub_num)
    filt = raw_from_layout(layout.derivatives['clean'], subject=sub_pad,
                           extension='.edf', desc='clean', preload=True)

    mne.viz.use_3d_backend('notebook')
    # %%
    plot_subj(filt, sub)
    # plot_on_average(filt, sub='D29')
    # plot_gamma(raw)
    # head_to_mni(raw, sub)
    # trans = mne.coreg.estimate_head_mri_t(sub, subj_dir)
    # mne.bem.make_watershed_bem(sub, subj_dir,
    # brainmask="../mri/brainmask.mgz")
    # fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=sub,
    #                              subjects_dir=subj_dir, dig=True,
    #                              show_axes=True)
    # # %%
    # T1_path = layout.get(return_type="path", subject=sub_pad,
    #                      extension="nii.gz")[0]
    # CT_path = T1_path.path.replace("T1w.nii.gz", "CT.nii.gz")
    # # filt = mne.io.read_raw_fif("D24_filt_ieeg.fif")
    # CT_aligned = allign_CT(T1_path, CT_path, sub)
    # subj_trans = mne.coreg.estimate_head_mri_t(sub, subjects_dir=subj_dir)
    # gui = mne.gui.locate_ieeg(raw.info, subj_trans, CT_aligned, subject=sub,
    #                           subjects_dir=subj_dir, verbose=10)
