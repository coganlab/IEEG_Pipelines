import logging
import os.path as op
from os import PathLike as PL
from os import walk
from re import match
from typing import Union, List, Tuple, Dict, TypeVar

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile
from dipy.align import resample
from joblib import cpu_count
from mne_bids import read_raw_bids, BIDSPath

HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
BIDS_root = op.join(LAB_root, "BIDS-1.3_Phoneme_sequencing", "BIDS")
PathLike = TypeVar("PathLike", str, PL)
RunDict = Dict[int, mne.io.Raw]
SubDict = Dict[str, RunDict]


def find_dat(folder: PathLike) -> Tuple[PathLike, PathLike]:
    cleanieeg = None
    ieeg = None
    for root, _, files in walk(folder):
        for file in files:
            if match(r".*cleanieeg\.dat.*", file):
                cleanieeg = op.join(root, file)
            elif match(r".*ieeg\.dat.*", file):
                ieeg = op.join(root, file)
            if ieeg is not None and cleanieeg is not None:
                return ieeg, cleanieeg
    raise FileNotFoundError("Not all .dat files were found:")


def line_filter(data: mne.io.Raw) -> mne.io.Raw:
    if not data.preload:
        data.load_data()
    filt = data.copy().notch_filter(None,  # (60, 120, 180, 240),
                                    # method='fir',
                                    # fir_window='blackman',
                                    # pad='reflect',
                                    method='spectrum_fit',
                                    mt_bandwidth=5.0,
                                    filter_length='20s',
                                    p_value=0.1,  # only used if freqs=None
                                    verbose=10,
                                    n_jobs=cpu_count() - 1)
    # make njobs 'cuda' with a gpu if method is 'fir'
    return filt


def mt_filt(data: mne.io.Raw):  # TODO: make your own filter
    """

    Steps:
    1. psd_multitaper
    2. f-test pre-defined freq range for max power using mne.stats.permutation_cluster_test (or just max)
    3. fft extract and sum frequencies found in max
    4. subtract frequencies in the time domain (signal - extracted_signal)

    """
    f, ax = plt.subplots()
    psds, freqs = mne.time_frequency.psd_multitaper(data, fmax=250, n_jobs=cpu_count(), verbose=10)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0)
    psds_std = psds.std(0)
    ax.plot(freqs, psds_mean, color='k')
    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                    color='k', alpha=.5)
    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')


def get_strongest_line(array: np.ndarray):
    pass


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


def allign_mri(t1_path: PathLike, ct_path: PathLike, my_raw: mne.io.Raw,
               sub_id: str, subj_dir=None):
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
    subj_trans = mne.coreg.estimate_head_mri_t(sub_id, subjects_dir=subj_dir)
    gui = mne.gui.locate_ieeg(my_raw.info, subj_trans, CT_aligned,
                              subject=sub_id, subjects_dir=subj_dir)


def bidspath_from_layout(layout: BIDSLayout, **kwargs) -> BIDSPath:
    my_search: list = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file, "
                                "try adding more search terms")
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    my_search: BIDSFile = my_search[0]
    entities = my_search.get_entities()
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


def raw_from_layout(layout: BIDSLayout, subject: str,
                    run: Union[List[int], int] = None) -> mne.io.Raw:
    kwargs = dict(subject=subject)
    if run:
        kwargs["run"] = run
    runs = layout.get(return_type="id", target="run", **kwargs)
    raw = []
    for r in runs:
        BIDS_path = bidspath_from_layout(layout, subject=subject, run=r,
                                         extension=".edf")
        raw.append(read_raw_bids(bids_path=BIDS_path))
        raw[-1].load_data()
    whole_raw = mne.concatenate_raws(raw)
    return whole_raw


def open_dat_file(file_path: str, channels: Union[str, List[str], int],
                  sfreq: int = 2048, types: str = "seeg") -> mne.io.RawArray:
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")
    channels.remove("Trigger")
    array = np.reshape(data, [len(channels), -1], order='F')
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array, info)
    return raw


def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)


def filt_main(layout: BIDSLayout,
              subjects: Union[str, List[str]] = None,
              runs: Union[int, List[int]] = None
              ) -> Tuple[SubDict, SubDict]:
    data = dict()
    raw_data = dict()
    subjects: List[str] = layout.get(return_type="id", target="subject",
                                     subject=subjects)
    for sub_id in subjects:
        raw_data[sub_id] = dict()
        data[sub_id] = dict()
        runs: List[int] = layout.get(return_type="id", target="run",
                                     subject=sub_id, run=runs)
        for run in runs:
            raw_data[sub_id][run] = raw_from_layout(layout, sub_id, run)
            data[sub_id][run] = line_filter(raw_data[sub_id][run])
    return raw_data, data


def filt_main_2(layout: BIDSLayout,
                subjects: Union[str, List[str]] = None,
                runs: Union[int, List[int]] = None):
    """A function that runs through all the subjects that filters then saves

    """
    subjects: List[str] = layout.get(return_type="id", target="subject",
                                     subject=subjects)
    for sub_id in subjects:
        runs: List[int] = layout.get(return_type="id", target="run",
                                     subject=sub_id, run=runs)
        for run in runs:
            raw_data = raw_from_layout(layout, sub_id, run)
            filt_data = line_filter(raw_data)
            save_name = "{}_filt_run-{}_ieeg.fif"
            filt_data.save(op.join(LAB_root, "Aaron_test", save_name))


def figure_compare(raw: List[mne.io.Raw], labels: List[str], avg: bool = True):
    for title, data in zip(labels, raw):
        fig = data.plot_psd(fmax=250, average=avg, n_jobs=cpu_count())
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large',
                     weight='bold')
        add_arrows(fig.axes[:2])


if __name__ == "__main__":
    logging.basicConfig(filename="Information.log", filemode="w",
                        level=logging.INFO)
    # TASK = "Phoneme_Sequencing"
    # SUB = "D22"
    # D_dat_raw, D_dat_filt = find_dat(op.join(LAB_root, "D_Data", TASK, SUB))
    layout = BIDSLayout(BIDS_root)
    filt_main_2(layout)
    # raw = raw_from_layout(layout, "D0022", [1, 2, 3, 4])
    # filt_dat = open_dat_file(D_dat_filt, raw.copy().ch_names)
    # raw_dat = open_dat_file(D_dat_raw, raw.copy().ch_names)
    # raw.load_data()
    # filt = mne.io.read_raw_fif("D22_filt_ieeg.fif")
    # raw_dat, dat = filt_main(layout, "D0028", 1)
    # data = [raw_dat, filt_dat, raw, filt]
    # figure_compare(data, ['Un',  '', "BIDS Un", "BIDS "])
    # T1_path = layout.get(return_type="path", subject="D0022", type="T1w")[0]
    # CT_path = layout.get(return_type="path", subject="D0022", type="CT")[0]
    # allign_mri(T1_path, CT_path, filt, "D0028")
