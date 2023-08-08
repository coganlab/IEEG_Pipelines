import os.path as op
from collections import OrderedDict
import csv
from functools import singledispatch

import matplotlib
from mne.viz import Brain
import matplotlib.patheffects as path_effects

import mne
import nibabel as nib
import numpy as np

from ieeg import Signal, PathLike

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt  # noqa: E402


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


def imshow_mri(data, img: nib.spatialimages.SpatialImage,
               vox: tuple[int, int, int], xyz: dict, suptitle: str = ""):
    """Show an MRI slice with a voxel annotated.

    Parameters
    ----------
    data : np.ndarray
        The data to plot
    img : nib.spatialimages.SpatialImage
        The image to plot
    vox : tuple[int, int, int]
        The voxel to annotate
    xyz : dict
        The xyz coordinates of the voxel
    suptitle : str, optional
        The title of the plot, by default ""

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        The figure and axes of the plot
    """
    i, j, k = vox
    fig, ax = plt.subplots(1, figsize=(6, 6))
    codes = nib.orientations.aff2axcodes(img.affine)
    # Figure out the title based on the code of this axis
    ori_slice = dict(
        P="Coronal", A="Coronal", I="Axial", S="Axial", L="Sagittal",
        R="Sagittal"
    )
    ori_names = dict(
        P="posterior", A="anterior", I="inferior", S="superior", L="left",
        R="right"
    )
    title = ori_slice[codes[0]]
    ax.imshow(data[i], vmin=10, vmax=120, cmap="gray", origin="lower")
    ax.axvline(k, color="y")
    ax.axhline(j, color="y")
    for kind, coords in xyz.items():
        annotation = "{}: {}, {}, {} mm".format(kind,
                                                *np.round(coords).astype(int))
        text = ax.text(k, j, annotation, va="baseline", ha="right",
                       color=(1, 1, 0.7))
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ]
        )
    # reorient view so that RAS is always rightward and upward
    x_order = -1 if codes[2] in "LIP" else 1
    y_order = -1 if codes[1] in "LIP" else 1
    ax.set(
        xlim=[0, data.shape[2] - 1][::x_order],
        ylim=[0, data.shape[1] - 1][::y_order],
        xlabel=f"k ({ori_names[codes[2]]}+)",
        ylabel=f"j ({ori_names[codes[1]]}+)",
        title=f"{title} view: i={i} ({ori_names[codes[0]]}+)",
    )
    fig.suptitle(suptitle)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
    return fig


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
        subj_dir = op.join(HOME, "Box", "ECoG_Recon")
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


def plot_on_average(sigs: Signal | str | list[Signal | str],
                    subj_dir: PathLike = None, rm_wm: bool = True,
                    picks: list[int | str] = None, surface: str = 'pial',
                    hemi: str = 'split', color: matplotlib.colors = (1, 1, 1),
                    size: float = 0.35, fig: Brain = None) -> Brain:
    """Plots the signal on the average brain

    Takes a signal instance or list of signal instances and plots them on the
    fsaverage brain.


    Parameters
    ----------
    sigs : Union[Signal, list[Signal]]
        The signal(s) to plot
    subj_dir : PathLike, optional
        The subjects directory, by default LAB_root / 'ECoG_Recon'
    rm_wm : bool, optional
        Whether to remove white matter electrodes, by default True
    picks : list[int | str], optional
        The channels to plot, by default None
    surface : str, optional
        The surface to plot on, by default 'pial'
    hemi : str, optional
        The hemisphere to plot, by default 'split'
    color : matplotlib.colors, optional
        The color to plot, by default (1, 1, 1)
    size : float, optional
        The size of the markers, by default 0.35
    fig : Brain, optional
        The figure to plot on, by default None

    Returns
    -------
    Brain
        The figure brain object
    """

    subj_dir = get_sub_dir(subj_dir)
    if fig is None:
        fig = Brain('fsaverage', subjects_dir=subj_dir, cortex='low_contrast',
                    alpha=0.6, background='grey', surf=surface, hemi=hemi)

    if isinstance(sigs, (Signal, mne.Info)):
        sigs = [sigs]
    if isinstance(sigs, list):
        sigs = {get_sub(v): v for v in sigs}

    for subj, inst in sigs.items():

        if isinstance(inst, mne.Info):
            new = inst.copy()
        elif isinstance(inst, Signal):
            new = inst.info.copy()
        elif isinstance(inst, str):
            new = subject_to_info(inst)
        else:
            raise TypeError(type(inst))

        to_fsaverage = mne.read_talxfm(subj, subj_dir)
        to_fsaverage = mne.transforms.Transform(fro='head', to='mri',
                                                trans=to_fsaverage['trans'])

        these_picks = range(len(new.ch_names))
        if isinstance(picks, (tuple, list)):
            if len(picks) == 0:
                continue
            elif isinstance(picks[0], int):
                these_picks = [new.ch_names[pick] for pick in these_picks if
                               pick in picks]
                picks = [p - len(new.ch_names) for p in
                         picks[len(these_picks):]]
            elif isinstance(picks[0], str):
                these_picks = [s[6:] for s in picks if s[:5] in
                               new['subject_info']['his_id']]
        elif picks is not None:
            raise TypeError(picks)

        if len(these_picks) == 0:
            continue

        if rm_wm:
            these_picks = pick_no_wm(these_picks, gen_labels(
                new, subj, subj_dir, new.ch_names))

        if len(these_picks) == 0:
            continue

        # plot the data
        plot_subj(new, subj_dir, these_picks, False, fig=fig,
                  trans=to_fsaverage, color=color, size=size,
                  labels_every=None)

    return fig


def pick_no_wm(picks: list[str], labels: OrderedDict[str, list[str]]):
    """Picks the channels that are not in the white matter

    Parameters
    ----------
    picks : list[str | int]
        The channels to pick from
    labels : dict[str | int, list[str]]
        The labels for each channel

    Returns
    -------
    list[str | int]
        The channels that are not in the white matter
    """

    # remove 'Unknown' values from label lists
    for k, v in labels.items():
        while 'Unknown' in v:
            labels[k].remove('Unknown')
            v = labels[k]

    # remove corresponding picks with either 'White-Matter' in the left most
    # entry or empty lists
    if isinstance(picks[0], int):
        picks = [list(labels.keys())[p] for p in picks]
    picks = [p for p in picks if labels[p]]
    picks = [p for p in picks if 'White-Matter' not in labels[p][0]]
    return picks


def get_sub(inst: Signal | mne.Info) -> str:
    """Gets the subject from the instance

    Parameters
    ----------
    inst : Signal
        The instance to get the subject from

    Returns
    -------
    str
        The subject"""
    if isinstance(inst, Signal):
        inst = inst.info
    return "D" + str(int(inst['subject_info']['his_id'][5:]))


def plot_subj(inst: Signal | mne.Info | str, subj_dir: PathLike = None,
              picks: list[str | int] = None, no_wm: bool = False,
              labels_every: int | None = 8, surface: str = 'pial',
              hemi: str = 'split', fig: Brain = None,
              trans=None, color: matplotlib.colors = (1, 1, 1),
              size: float = 0.3, title: str = None) -> Brain:
    """Plots the electrodes on the subject's brain

    Parameters
    ----------
    inst : Signal | mne.Info | str
        The subject to plot
    subj_dir : PathLike, optional
        The subjects directory, by default HOME / 'Box' / 'ECoG_Recon'
    picks : list[str | int], optional
        The channels to plot, by default all
    no_wm : bool, optional
        Whether to remove the white matter channels, by default False
    labels_every : int, optional
        How often to label the channels, by default 8
    fig : Brain, optional
        The figure to plot on, by default None
    surface : str, optional
        The surface to plot, by default 'pial'
    hemi : str, optional
        The hemisphere to plot, by default 'split'
    trans: mne.transforms.Transform, optional
        The transformation to apply, by default None
    color : matplotlib.colors, optional
        The color of the electrodes, by default (1,1,1)
    size : float, optional
        The size of the electrodes, by default 0.35
    title : string, optional
        Title the plot

    Returns
    -------
    Brain
        The brain plot
    """

 

    if isinstance(inst, Signal):
        info = inst.info
        sub = get_sub(info)
    elif isinstance(inst, mne.Info):
        info = inst
        sub = get_sub(info)
    elif isinstance(inst, str):
        info = subject_to_info(inst, subj_dir)
        sub = inst
    else:
        raise TypeError(
            f"inst must be Signal, mne.Info, or str, not {type(inst)}")
    

    if subj_dir is None:
        subj_dir = get_sub_dir(subj_dir)
    if trans is None:
        trans = mne.transforms.Transform(fro='head', to='mri')
    if fig is None:
        fig = Brain(sub, subjects_dir=subj_dir, cortex='low_contrast',
                    alpha=0.5,
                    background='grey', surf=surface, hemi=hemi)
    # Set the title if provided
    if title is not None:
        mne.viz.set_3d_title(fig, title, size=40)
    if picks is None:
        picks = info.ch_names
    if no_wm:
        picks = pick_no_wm(picks,
                           gen_labels(info, sub, subj_dir, info.ch_names))
    if isinstance(picks[0], str):
        picks = mne.pick_channels(info.ch_names, picks)

    info: mne.Info = mne.pick_info(info, picks)
    # fig.add_sensors(info, trans)
    montage = info.get_montage()
    force2frame(montage, trans.from_str)
    montage.apply_trans(trans)
    pos = montage.get_positions()['ch_pos']

    # Default montage positions are in m, whereas plotting functions assume mm
    left = [p * 1000 for k, p in pos.items() if k.startswith('L')]
    right = [p * 1000 for k, p in pos.items() if k.startswith('R')]

    if left:
        fig.add_foci(np.vstack(left), hemi='lh', color=color,
                     scale_factor=size)
    if right:
        fig.add_foci(np.vstack(right), hemi='rh', color=color,
                     scale_factor=size)

    if labels_every is not None:
        settings = dict(shape=None, always_visible=True, text_color=(0, 0, 0),
                        bold=False)
        _add_labels(fig, info, sub, pos, labels_every, hemi,
                    (left, right), **settings)

    return fig


def _add_labels(fig, info, sub, pos, every, hemi, lr, **kwargs):

    picks = info.ch_names

    names = picks[slice(every - 1, info['nchan'], every)]

    if hemi == 'split':
        for hems, positions in zip(range(2), lr):
            if not positions:
                continue
            pos = positions[slice(every - 1, info['nchan'], every)]
            plt_names = [f'{sub}-{n}' for n in names if
                         n.startswith(['L', 'R'][hems])]
            fig.plotter.subplot(0, hems)
            fig.plotter.add_point_labels(pos, plt_names, **kwargs)
    else:
        plt_names = [f'{sub}-{n}' for n in names]
        positions = np.array([pos[name] for name in names]) * 1000
        fig.plotter.add_point_labels(positions, plt_names, **kwargs)


def subject_to_info(subject: str, subjects_dir: PathLike = None,
                    ch_types: str = "seeg", sfreq: int = 2000) -> mne.Info:
    """Gets the info for a subject from the subjects directory

    Parameters
    ----------
    subject : str
        The subject to get the info for
    subjects_dir : PathLike, optional
        The subjects directory, by default HOME / 'Box' / 'ECoG_Recon'
    ch_types : str, optional
        The channel type, by default "seeg"
    sfreq : int, optional
        The sampling frequency, by default 2000

    Returns
    -------
    mne.Info
        The info for the subject
    """
    subjects_dir = get_sub_dir(subjects_dir)
    elec_file = op.join(subjects_dir, subject, 'elec_recon',
                        subject + '_elec_locations_RAS_brainshifted.txt')
    elecs = dict()
    with open(elec_file, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            line = row[0].split(" ")
            elecs["".join(line[0:2])] = tuple(
                float(n) / 1000 for n in line[2:5])
    info = mne.create_info(list(elecs.keys()), sfreq, ch_types)
    montage = mne.channels.make_dig_montage(elecs, nasion=(0, 0, 0),
                                            coord_frame='ras')
    info.set_montage(montage)
    return info


@singledispatch
def force2frame(montage: mne.channels.DigMontage, frame: str):
    """Forces the montage to be in the specified frame

    Parameters
    ----------
    montage : mne.channels.DigMontage
        The montage to force
    frame : str, optional
        The frame to force to, by default 'mri'
    """

    settings = dict(fro=montage.get_positions()['coord_frame'],
                    to=frame, trans=np.eye(4))
    # current subjects are in 'mri' space, even though it says head
    if not settings['fro'] == frame:
        trans = mne.transforms.Transform(**settings)
        montage.apply_trans(trans)


@force2frame.register
def _(info: mne.Info, frame: str):
    montage = info.get_montage()
    force2frame(montage, frame)
    info.set_montage(montage)


def gen_labels(info: mne.Info, sub: str = None, subj_dir: PathLike = None,
               picks: list[str | int] = None) -> OrderedDict[str, list[str]]:
    """Generates the labels for the electrodes

    Parameters
    ----------
    info : mne.Info
        The subject to get the labels for
    subj_dir : PathLike, optional
        The subjects directory, by default None
    picks : list[str | int], optional
        The channels to plot, by default None

    Returns
    -------
    dict[str, list]
        The labels for the electrodes
    """

    sub = get_sub(info) if sub is None else sub
    subj_dir = get_sub_dir(subj_dir)
    montage = info.get_montage()
    force2frame(montage, 'mri')
    aseg = 'aparc.a2009s+aseg'  # parcellation/anatomical segmentation atlas
    labels, colors = mne.get_montage_volume_labels(
        montage, sub, subjects_dir=subj_dir, aseg=aseg)

    new_labels = OrderedDict()
    if picks is not None:
        for i, key in enumerate(labels.keys()):
            if any((i in picks, key in picks)):
                new_labels[key] = labels[key]

    return new_labels


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
    sub_num = 59
    layout = get_data(TASK, root=LAB_root)
    subj_dir = op.join(LAB_root, "ECoG_Recon_Full")
    sub_pad = "D" + str(sub_num).zfill(4)
    # sub = "D{}".format(sub_num)

    filt = raw_from_layout(layout.derivatives['clean'], subject=sub_pad,
                           extension='.edf', desc='clean', preload=False)

    ##
    brain = plot_subj(filt)
    # plot_on_average(filt)
    # plot_gamma(raw)
