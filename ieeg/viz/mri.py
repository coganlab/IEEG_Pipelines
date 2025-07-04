import csv
import os.path as op
from collections import OrderedDict, namedtuple
from collections.abc import Iterable, Sequence
from functools import singledispatch

import mne
import nibabel as nib
import numpy as np
import pandas as pd
from mne.viz import Brain

from ieeg import PathLike, Signal
from ieeg.io import get_elec_volume_labels
from ieeg.viz import _qt_backend, parula
from pyvistaqt import BackgroundPlotter

_qt_backend()

import matplotlib  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
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
        The subjects directory, by default LAB_root / 'ECoG_Recon'

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
        ax.plot(x_line + x, gamma_power[i] + y, linewidth=0.5,
                color=color)


def get_grey_matter(subjects: Sequence[str]) -> set[str]:
    """Gets the grey matter channels for a list of subjects

    Parameters
    ----------
    subjects : Sequence[str]
        The subjects to get the grey matter channels for

    Returns
    -------
    set[str]
        The grey matter channels
    """
    grey_matter = set()
    for i, subj in enumerate(subjects):
        info = subject_to_info(get_sub(subj))
        parcel_label = gen_labels(info, get_sub(subj))
        subj_grey_matter = pick_no_wm(info.ch_names, parcel_label)

        # get the indices of channels in info that are not in grey_matter
        grey_matter |= {f"{subj}-{ch}" for ch in info.ch_names
                        if ch in subj_grey_matter}
    return grey_matter


def plot_on_average(sigs: Signal | str | mne.Info | list[Signal | str, ...],
                    subj_dir: PathLike = None, rm_wm: bool = False,
                    picks: list[int | str, ...] = None, surface: str = 'pial',
                    hemi: str = 'split', color: matplotlib.colors = (1, 1, 1),
                    size: float = 0.35, fig: Brain = None,
                    label_every: int = None, background: str = 'white',
                    units: str = 'm', transparency: float = 0.6,
                    average: str = 'fsaverage', show: bool = True) -> Brain:
    """Plots the signal on the average brain

    Takes a signal instance or list of signal instances and plots them on the
    fsaverage brain.

    Parameters
    ----------
    sigs : Union[Signal, list[Signal]]
        The signal(s) to plot. If a list, all electrodes will be the same
        color. If a single signal, each electrode will be a different color.
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
        The color to plot, by default (1, 1, 1). If None, each electrode group
        will be a different color.
    size : float, optional
        The size of the markers, by default 0.35
    fig : Brain, optional
        The figure to plot on, by default None
    label_every : int, optional
        How often to label the channels, by default None
    background: str, optional
        Background color
    units: str, optional
        Units of the electrodes
    transparency: float, optional
        Transparency of the brain
    average: str, optional
        The average brain to plot on, by default 'fsaverage'
    show: bool, optional
        Whether to show the figure, by default True

    Returns
    -------
    Brain
        The figure brain object
    """

    subj_dir = get_sub_dir(subj_dir)
    if fig is None:
        fig = Brain(average, subjects_dir=subj_dir, cortex='low_contrast',
                    alpha=transparency, background=background, surf=surface,
                    hemi=hemi, units=units, show=show)
    if picks is not None:
        if len(picks) == 0:
            return fig

    if isinstance(sigs, (Signal, mne.Info)):
        sigs = [sigs]
        color = None
    if isinstance(sigs, Iterable):
        sigs = {get_sub(v): v for v in sigs}

    all_channel_name = []
    for subj, inst in sigs.items():
        if isinstance(inst, mne.Info):
            new = inst.copy()
        elif isinstance(inst, Signal):
            new = inst.info.copy()
        elif isinstance(inst, str):
            new = subject_to_info(subj)
            new['subject_info'] = dict(his_id=f"sub-{inst}")
        else:
            raise TypeError(type(inst))
        sigs[subj] = new
        for ch in new.ch_names:
            all_channel_name.append(f"{subj}-{ch}")

    if picks is None:
        picks = all_channel_name
    elif isinstance(picks[0], int):
        picks = [all_channel_name[p] for p in picks]
    elif isinstance(picks[0], str):
        if len(sigs) == 1 and not picks[0].startswith(list(sigs.keys())[0]):
            picks = [subj + '-' + p for p in picks]
        picks_in = [p in all_channel_name for p in picks]
        assert all(picks_in), (f"Channel not found: "
                               f"{picks[picks_in.index(False)]}")
    else:
        raise TypeError(f"picks must be list of str or int, not "
                        f"{type(picks[0])}")

    default_c = parula.mat_colors.copy()
    for subj, new in sigs.items():

        to_fsaverage = mne.read_talxfm(subj, subj_dir)
        if average == 'fsaverage':
            trans = mne.transforms.Transform(fro='head', to='mri',
                                             trans=to_fsaverage['trans'])
        else:
            from_average = mne.read_talxfm(average, subj_dir)
            to_average = np.dot(np.linalg.inv(from_average['trans']),
                                to_fsaverage['trans'])
            trans = mne.transforms.Transform(fro='head', to='mri',
                                             trans=to_average)

        pick_gen = (p.split('-') for p in picks)
        these_picks = [p[1] for p in pick_gen if p[0] == subj]

        if rm_wm:
            these_picks = pick_no_wm(these_picks, gen_labels(
                new, subj, subj_dir, picks=new.ch_names))

        if len(these_picks) == 0:
            continue

        # select colors
        if color is None and len(sigs) > 1:
            this_color = []
            p_int = [new.ch_names.index(p) for p in these_picks]
            groups = _group_channels(mne.pick_info(new, p_int))
            n_groups = len(set(groups.values()))
            while len(this_color) < n_groups:
                this_color += [default_c.pop(0)]
        elif np.isscalar(color) or color is None:
            this_color = color
        elif len(color) == len(picks):
            this_color = [color[picks.index(subj + '-' + p)]
                          for p in these_picks]
        else:
            this_color = color

        if not np.isscalar(size):
            size = list(size)
            this_size = [size.pop(0) for p in these_picks]
        else:
            this_size = [size] * len(these_picks)

        # plot the data
        plot_subj(new, subj_dir, these_picks, False, fig=fig,
                  trans=trans, color=this_color, size=this_size,
                  labels_every=label_every, hemi=hemi, background=background,
                  show=show)

    return fig


def pick_no_wm(picks: list[str], labels: OrderedDict[str: list[str]]) -> list:
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
    bad_words = ('Unknown', 'unknown', 'hypointensities', 'White-Matter')
    if len(picks) == 0:
        return []
    # remove corresponding picks with either 'White-Matter' in the left most
    # entry or empty lists
    if isinstance(picks[0], int):
        picks = [list(labels.keys())[p] for p in picks]
    picks = [p for p in picks if not any(w in labels[p] for w in bad_words)]
    return picks


def get_sub(inst: Signal | mne.Info | str) -> str:
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
    elif isinstance(inst, str):
        return f"{inst[0]}{int(inst[1:])}"
    out_str = inst['subject_info']['his_id'][4:]
    if len(out_str) == 1:
        return out_str
    return out_str[0] + str(int("".join(s for s in out_str if s.isdigit())))


# TODO: figure out why elec positions are only correct in meters not mm


def plot_subj(inst: Signal | mne.Info | str, subj_dir: PathLike = None,
              picks: list[str | int] = None, no_wm: bool = False,
              labels_every: int | None = 8, surface: str = 'pial',
              hemi: str = 'both', fig: Brain = None,
              trans=None, color: matplotlib.colors = None,
              size: float = 0.35, show: bool = True, background: str = 'white',
              title: str = None, units: str = 'm', transparency: float = 0.5,
              cortex: str = 'low_contrast') -> Brain:
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
    show : bool, optional
        Whether to show the figure, by default True
    background: str, optional
        Background color
    title : string, optional
        Title the plot
    units: str, optional
        Units of the electrodes
    transparency: float, optional
        Transparency of the brain
    cortex: str, optional
        The cortex to plot

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
        fig = Brain(sub, subjects_dir=subj_dir, cortex=cortex,
                    alpha=transparency, background=background, surf=surface,
                    hemi=hemi, show=show, units=units)

    if isinstance(color, str):
        color = matplotlib.colors.to_rgb(color)

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

    groups = _group_channels(info)
    n_groups = len(set(groups.values()))
    vals = list(groups.values())
    if color is None:
        color = []
        for i in range(n_groups):
            start, end = vals.index(i), len(vals) - vals[::-1].index(i)
            col = parula.mat_colors[i]
            color += [col] * (end - start)
    elif np.isscalar(color) or isinstance(color, tuple):
        color = [color] * len(vals)
    else:
        color = list(color)

    # fig.add_sensors(info, trans)
    montage = info.get_montage()
    force2frame(montage, trans.from_str)
    montage.apply_trans(trans)
    pos = {k: v for k, v in montage.get_positions()['ch_pos'].items()}

    # Default montage positions are in m, whereas plotting functions assume mm
    left = {}
    right = {}
    if np.isscalar(size):
        size = [size] * len(pos)
    else:
        size = list(size)

    lsize = []
    rsize = []
    lcolor = []
    rcolor = []
    for k, v in pos.items():
        if k.startswith('L'):
            left[k] = v
            lsize.append(size.pop(0))
            lcolor.append(color.pop(0))
        elif k.startswith('R'):
            right[k] = v
            rsize.append(size.pop(0))
            rcolor.append(color.pop(0))
        elif v[0] < 0:
            left[k] = v
            lsize.append(size.pop(0))
            lcolor.append(color.pop(0))
        else:
            right[k] = v
            rsize.append(size.pop(0))
            rcolor.append(color.pop(0))

    if left and hemi != 'rh':
        _add_electrodes(fig, 'lh', np.vstack(list(left.values())),
                        lcolor, lsize)
    if right and hemi != 'lh':
        _add_electrodes(fig, 'rh', np.vstack(list(right.values())),
                        rcolor, rsize)

    if labels_every is not None:
        settings = dict(shape=None, always_visible=True,
                        text_color=fig._fg_color, bold=False)
        _add_labels(fig, info, sub, labels_every, hemi,
                    (left, right), **settings)

    return fig


def electrode_gradient(subjects: list[Signal | str, ...], W: np.ndarray,
                       idx: list[int], colors: list,
                       mode: str = 'both', max_size: float = 2,
                       fig_dims: tuple[int, int] = None) -> None:
    """Plots the electrodes with a gradient of colors

    """
    if fig_dims is None:
        min_size = int(np.ceil(np.sqrt(W.shape[0])))
        fig_dims = (int(np.ceil(np.sqrt(W.shape[0] / min_size))), min_size)
    plotter = BackgroundPlotter(shape=fig_dims)
    scale = W.copy()
    scale[scale > max_size] = max_size

    if mode in ['size', 'both']:
        size = scale / 2
    else:
        size = np.ones_like(scale)

    if mode in ['fade', 'both']:
        colors = _create_color_alpha_matrix(colors, scale / scale.max())

    for i in range(W.shape[0]):
        j, k = divmod(i, fig_dims[1])
        plotter.subplot(j, k)
        brain = plot_on_average(subjects, picks=list(idx), size=size[i],
                                hemi='both', color=colors[i], show=False,
                                transparency=0.15)
        for actor in brain.plotter.actors.values():
            plotter.add_actor(actor, reset_camera=False)
        plotter.camera = brain.plotter.camera
        plotter.camera_position = brain.plotter.camera_position
    plotter.link_views()


def _create_color_alpha_matrix(colors: list, alphas: np.ndarray) -> np.ndarray:
    n = len(colors)
    m = alphas.shape[1]
    color_alpha_matrix = np.empty((n, m), dtype=object)  # 4 for RGBA

    for i, color in enumerate(colors):
        rgba_color = matplotlib.colors.to_rgba(color)
        for j, alpha in enumerate(alphas[i]):
            color_alpha_matrix[i, j] = rgba_color[:3] + (alpha,)

    return color_alpha_matrix


def _add_electrodes(fig: mne.viz.Brain, hemi: str,
                    pos: np.ndarray, colors: matplotlib.colors = None,
                    size: float | list = 0.35):

    if not np.isscalar(size) and len(colors[0]) == 4:
        assert len(size) == len(pos), "Size must be the same length as vals"
        assert len(colors) == len(pos), ("Colors must be the same length as"
                                         " vals")
        for i, p in enumerate(pos):
            fig.add_foci(p, hemi=hemi, color=colors[i][:3],
                         scale_factor=size[i], alpha=colors[i][3])
    elif len(colors[0]) == 4:
        assert len(colors) == len(
            pos), "Colors must be the same length as vals"
        for i, p in enumerate(pos):
            fig.add_foci(p, hemi=hemi, color=colors[i][:3],
                         scale_factor=size, alpha=colors[i][3])
    elif not np.isscalar(size):
        while len(colors) < len(size):
            colors.append(colors[-1])
        assert len(size) == len(pos), "Size must be the same length as vals"
        for i, p in enumerate(pos):
            fig.add_foci(p, hemi=hemi, color=colors[i],
                         scale_factor=size[i])
    else:
        for i, p in enumerate(pos):
            fig.add_foci(p, hemi=hemi, color=colors, scale_factor=size)


def _group_channels(info, groups: dict = None) -> dict:
    """Automatically find a group based on the name of the channel."""
    if groups is not None:
        for name in info.ch_names:
            if name not in groups.keys():
                raise ValueError(f"{name} not found in ``groups``")
        return groups

    i = 0
    groups = dict()
    base_names = dict()
    for name in info.ch_names:
        # strip all numbers from the name
        base_name = "".join(
            [
                letter
                for letter in name
                if not letter.isdigit() and letter != " "
            ]
        )
        if base_name in base_names:
            # look up group number by base name
            groups[name] = base_names[base_name]
        else:
            groups[name] = i
            base_names[base_name] = i
            i += 1
    return groups


def _add_labels(fig, info, sub, every, hemi, lr, **kwargs):
    if len(info.ch_names) < every:
        return
    names = info.ch_names[slice(every - 1, None, every)]
    if not hemi == 'both':
        for hems, pos in enumerate(lr):
            if (not pos) or \
                    (hemi == 'lh' and hems == 1) or \
                    (hemi == 'rh' and hems == 0):
                continue

            plt_names = filter(lambda x: x.startswith(['L', 'R'][hems]), names)
            plt_names = [f'{sub}-{n}' for n in plt_names]
            positions = np.array([pos[n.split("-")[1]] for n in plt_names])
            if hemi == 'split':
                fig.plotter.subplot(0, hems)
            fig.plotter.add_point_labels(positions, plt_names, **kwargs)
    else:
        pos = {}
        for hem in lr:
            if hem:
                pos.update(hem)
        plt_names = [f'{sub}-{n}' for n in names]
        positions = np.array([pos[name] for name in names])
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


def gen_labels(info: mne.Info, sub: str = None, subj_dir: str = None,
               atlas: str = ".a2009s", picks: list[str] = None
               ) -> OrderedDict[str, list[str]]:
    """Generates the labels for the electrodes

    Parameters
    ----------
    info : mne.Info
        The subject to get the labels for
    sub : str, optional
        The subject to get the labels for, by default None
    subj_dir : PathLike, optional
        The subjects directory, by default None
    picks : list[str | int], optional
        The channels to plot, by default None
    atlas : str, optional
        The atlas to use, by default ".a2009s"

    Returns
    -------
    dict[str, list]
        The labels for the electrodes
    """
    sub = get_sub(info) if sub is None else sub
    subj_dir = get_sub_dir(subj_dir)
    montage = info.get_montage()
    force2frame(montage, 'mri')
    # aseg = 'aparc.a2009s+aseg'  # parcellation/anatomical segmentation atlas
    labels = get_elec_volume_labels(sub, subj_dir, 10, atlas)
    if picks is None:
        picks = info.ch_names
    bad_words = ('Unknown', 'unknown', 'hypointensities', 'White-Matter')
    new_labels = OrderedDict()
    for p in picks:
        new_labels[p] = _pick_label(labels.T[p], 0.05, bad_words)
    return new_labels


def _pick_label(label: pd.Series, percent_thresh: float,
                bad_words: list[str] = ('Unknown', 'unknown',
                                        'hypointensities', 'White-Matter')):

    i = 2
    if label[0] not in bad_words:
        return label[0]
    while not ((not any(w in label[i] for w in bad_words)) and
               float(label[i + 1]) > percent_thresh):
        if (i + 2) >= len(label.T):  # end of labels
            return label[0]
        elif label[i + 2].isspace():  # empty label
            return label[0]
        else:
            i += 2
    return label[i]


def find_labels(info: mne.Info, sub: str = None, subj_dir: str = None,
                atlas: str = ".a2009s", hot_words: list[str] = None,
                pct_thresh: float = 0.1) -> OrderedDict[str, list[str]]:
    """Prioritize labels that contain hot_words, otherwise use _pick_label
     to find most probable GM

    Parameters
    ----------
    info : mne.Info
        The subject to get the labels for
    sub : str, optional
        The subject to get the labels for, by default None
    subj_dir : PathLike, optional
        The subjects directory, by default None
    atlas : str, optional
        The atlas to use, by default ".a2009s"
    hot_words : list[str], optional
    pct_thresh : float, optional

    Returns
    -------
    dict[str, list]
        The labels for the electrodes
    """
    sub = get_sub(info) if sub is None else sub
    subj_dir = get_sub_dir(subj_dir)
    labels = get_elec_volume_labels(sub, subj_dir, 10, atlas)
    chlist = info.ch_names
    new_labels = OrderedDict()
    for c in chlist:
        new_labels[c] = _find_label(labels.T[c], pct_thresh, hot_words)
    return new_labels


def _find_label(label: pd.Series, percent_thresh: float,
                hot_words: list[str] = None):

    i = 2
    if any(w in label[0] for w in hot_words):
        return label[0]
    while not ((any(w in label[i] for w in hot_words)) and (float(label[i + 1])
                                                            > percent_thresh)):
        # end of label or empty label
        if ((i + 2) >= len(label.T)) or (label[i + 2].isspace()):
            return _pick_label(label, 0.05)
        i += 2
    return label[i]


class Atlas:
    def __init__(self, subjects_dir: PathLike = None,
                 atlas: str = 'BNA_subregions.xlsx',
                 delim: str = ','):
        subjects_dir = get_sub_dir(subjects_dir)
        ref = pd.ExcelFile(op.join(subjects_dir, atlas))
        data = ref.parse(ref.sheet_names[0])
        self.entries = []
        self.abbreviations = {}
        lobe = None
        gyrus = None
        entry = namedtuple('Entry', ['lobe', 'gyrus',
                                     'subregion', 'MNI'])
        for row in data.iterrows():
            lobe = row[1]['Lobe'].split(' ')[0] if not isinstance(
                row[1]['Lobe'], float) else lobe
            gyrus = self.parse_abbrev(row[1]['Gyrus'], delim
                                      ) if not isinstance(
                row[1]['Gyrus'], float) else gyrus
            subregion = self.parse_abbrev(row[1].iloc[5], delim)
            lh = tuple(int(coord) for coord in row[1].iloc[7].split(delim))
            rh = tuple(int(coord) for coord in row[1].iloc[8].split(delim))
            self.entries.append(entry(lobe, gyrus, subregion,
                                      {'lh': lh, 'rh': rh}))

    def __getitem__(self, key):
        out = [entry for entry in self.entries if entry.subregion == key]
        if len(out) == 0:
            if key in self.abbreviations.values():
                key = [k for k, v in self.abbreviations.items() if v == key][0]
                return self[key]
            else:
                raise KeyError(f"Subregion {key} not found")
        elif len(out) == 1:
            return out[0]
        else:
            return out

    @property
    def gyri(self):
        gyri = (entry.gyrus for entry in self.entries)
        expanded = (self.abbreviations[g]
                    if g in self.abbreviations else g
                    for g in gyri)
        return set(expanded)

    @property
    def lobes(self):
        lobes = (entry.lobe for entry in self.entries)
        expanded = (self.abbreviations[lobe]
                    if lobe in self.abbreviations else lobe
                    for lobe in lobes)
        return set(expanded)

    def __repr__(self):
        num_gyri = len(self.gyri)
        num_lobes = len(self.lobes)
        num_subregions = len(self.entries)
        return (f"Atlas with {num_gyri} gyri, {num_lobes} lobes, and"
                f" {num_subregions} subregions")

    def parse_abbrev(self, abbrev: str, delim: str = ','):
        short, *long = abbrev.split(delim)
        self.abbreviations[short] = delim.join(long).lstrip().rstrip()
        return short


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
    TASK = "Phoneme_sequencing"
    sub_num = 35
    # layout = get_data(TASK, root=LAB_root)
    subj_dir = op.join(LAB_root, "..", "ECoG_Recon")
    sub_pad = "D" + str(sub_num).zfill(4)
    info = subject_to_info(f"D{sub_num}", subj_dir)
    labels = gen_labels(info, sub=f"D{sub_num}", subj_dir=subj_dir,
                        atlas=".BN_atlas")
    bn_atlas = Atlas()

    # sub = "D{}".format(sub_num)
    #
    # filt = raw_from_layout(layout.derivatives['clean'], subject=sub_pad,
    #                        extension='.edf', desc='clean', preload=False)
    #
    # #
    # electrode_gradient(["D5"], np.random.random((2, 48)),
    #                    list(range(48)),
    #                    [[1, 0, 0], [0, 1, 0]], mode='both')
    # sample_path = mne.datasets.sample.data_path()
    # subjects_dir = sample_path / "subjects"
    # plot_subj("D5")
    # colors = np.concatenate([np.array([[1,0,0]] * 48), (np.arange(48) / 48)[
    # :, None]], axis=1)
    # brain = plot_subj("D5", color=colors)
    # colors = np.concatenate([colors, np.random.random((124, 4))], axis=0)
    #
    # substring = "D35"
    # sublist = substring.split()
    #
    # elec_label_tofind = 'LPI6'
    # elec_list = list(labels.keys())
    # elec_picks = elec_list.index(elec_label_tofind)
    # plot_on_average(sublist, rm_wm=False, hemi='both', picks=[elec_picks])
    # plot_gamma(raw)
    # plot_on_average(["D22", "D28", "D64"],
    #                 picks=["D22-LPIF4", "D28-LPIO7", "D64-LAI6"],
    #                 label_every=1, hemi='lh', rm_wm=False, color='red',
    #                 size=1)
