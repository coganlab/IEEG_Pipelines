import os.path as op
from os import walk
from re import match
from typing import Union, List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from bids import BIDSLayout
from bids.layout import BIDSFile
from joblib import cpu_count
from mne_bids import read_raw_bids, BIDSPath

from utils import HOME, LAB_root, PathLike, figure_compare
from mri import allign_CT, show_brain, head_to_mni, plot_gamma

BIDS_root = op.join("..","..","..","..","Sentence_Rep", "BIDS")
RunDict = Dict[int, mne.io.Raw]
SubDict = Dict[str, RunDict]


def find_dat(folder: PathLike) -> Tuple[PathLike, PathLike]:
    cleanieeg = None
    ieeg = None
    for root, _, files in walk(folder):
        for file in files:
            if match(r".*cleanieeg\.dat.*", file):
                cleanieeg: PathLike = op.join(root, file)
            elif match(r".*ieeg\.dat.*", file):
                ieeg: PathLike = op.join(root, file)
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
                                    p_value=0.05,  # only used if freqs=None
                                    verbose=10,
                                    n_jobs=cpu_count())
    # make njobs 'cuda' with a gpu if method is 'fir'
    return filt


def mt_filt(data: mne.io.Raw):  # TODO: make your own filter
    """

    Steps:
    1. psd_multitaper
    2. f-test pre-defined freq range for max power using mne.stats.permutation
       _cluster_test (or just max)
    3. fft extract and sum frequencies found in max
    4. subtract frequencies in the time domain (signal - extracted_signal)

    """
    f, ax = plt.subplots()
    psds, freqs = mne.time_frequency.psd_multitaper(data, fmax=250,
                                                    n_jobs=cpu_count(),
                                                    verbose=10)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0)
    psds_std = psds.std(0)
    ax.plot(freqs, psds_mean, color='k')
    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                    color='k', alpha=.5)
    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')


def get_strongest_line(myarray: np.ndarray):
    pass


def bidspath_from_layout(layout: BIDSLayout, **kwargs: dict) -> BIDSPath:
    my_search: List[BIDSFile] = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file, "
                                "try adding more search terms")
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    found = my_search[0]
    entities = found.get_entities()
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


def raw_from_layout(layout: BIDSLayout, subject: str,
                    run: Union[List[int], int] = None) -> mne.io.Raw:
    kwargs: Dict[str, Any] = dict(subject=subject)
    if run:
        kwargs["run"] = run
    runs = layout.get(return_type="id", target="run", **kwargs)
    raw: List[mne.io.Raw] = []
    for r in runs:
        BIDS_path = bidspath_from_layout(layout, subject=subject, run=r,
                                         extension=".edf")
        new_raw = read_raw_bids(bids_path=BIDS_path)
        new_raw.load_data()
        raw.append(new_raw.copy())
        del new_raw
    whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    return whole_raw


def open_dat_file(file_path: str, channels: List[str],
                  sfreq: int = 2048, types: str = "seeg") -> mne.io.RawArray:
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")
    channels.remove("Trigger")
    array = np.reshape(data, [len(channels), -1], order='F')
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array, info)
    return raw


def retrieve_filt(sub: str,
                  runs: Union[List[int], int] = (1, 2, 3, 4)) -> mne.io.Raw:
    try:
        iter(runs)
    except TypeError:
        runs = [runs]
    if not isinstance(runs, list):
        runs = list(runs)
    filt = mne.io.read_raw_fif(
        op.join(LAB_root, "Aaron_test", "filt_phonemesequence",
                "{}_filt_run-{}_ieeg.fif".format(sub, 1)))
    del runs[0]
    for i in runs:
        f_name = op.join(LAB_root, "Aaron_test", "filt_phonemesequence",
                         "{}_filt_run-{}_ieeg.fif".format(sub, i))
        mne.concatenate_raws([filt, mne.io.read_raw_fif(f_name)])
    return filt


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


def filt_main_2(layout: BIDSLayout, subjects: Union[str, List[str]] = None):
    """A function that runs through all the subjects that filters then saves

    """
    if subjects is None:
        subjects: List[str] = layout.get(return_type="id", target="subject")
    for sub_id in subjects:
        runs: List[int] = layout.get(return_type="id", target="run",
                                     subject=sub_id)
        for run in runs:
            try:
                raw_data = raw_from_layout(layout, sub_id, run)
                filt_data = line_filter(raw_data)
                save_name = "{}_filt_run-{}_ieeg.fif".format(sub_id, run)
                filt_data.save(op.join(LAB_root, "Aaron_test",
                                    "filt_phonemesequence", save_name),
                            overwrite=False)
            except Exception:
                pass
        del runs


if __name__ == "__main__":
    log_filename = "output.log"  # op.join(LAB_root, "Aaron_test", "Information.log")
    mne.set_log_file(log_filename,
                     "%(levelname)s: %(message)s - %(asctime)s",
                     overwrite=True)
    mne.set_log_level("INFO")
    sub_num = 53
    sub_pad = "D00{}".format(sub_num)
    subject = "D{}".format(sub_num)
    layout = BIDSLayout(BIDS_root)
    raw = raw_from_layout(layout, sub_pad, 2)
    #raw.plot(n_channels=3,precompute=True, start=90)
    # filt = retrieve_filt(sub_pad, 1)
    """
    T1_path = layout.get(subject=sub_pad, extension="nii.gz")[0]
    CT_path = T1_path.path.replace("T1w.nii.gz", "CT.nii.gz")
    subjects_dir = op.join(LAB_root, "ECoG_Recon_Full")
    if subject == "D24":
        reg_affine = np.array([[1.00000,0.00000,0.00000,-5.69848],
                  [0.00000,1.00000,0.00000,25.40463],
                  [0.00000,0.00000,1.00000,63.92385],
                  [0.00000,0.00000,0.00000,1.00000]])
    elif subject == "D29":
        reg_affine = np.array([[ 0.99949355, -0.02990436,  0.01087993,  0.00315547],
                            [ 0.03083844,  0.99457727, -0.09932295, -0.03061648],
                            [-0.00785074,  0.09960816,  0.99499577, -0.00894423],
                            [ 0.,          0.,          0.,          1.        ]])
    else:
        reg_affine = None
    CT_aligned = allign_CT(T1_path, CT_path, reg_affine)
    subj_trans = mne.coreg.estimate_head_mri_t(subject, subjects_dir=subjects_dir)
    inv_trans = mne.coreg.invert_transform(mne.transforms.Transform("head", "head", subj_trans['trans']))
    mri = filt.copy()
    montage = mri.get_montage()
    montage.apply_trans(inv_trans)
    mri.set_montage(montage)
    mri.info = mne.preprocessing.ieeg.project_sensors_onto_brain(
        mri.info, subj_trans, subject, subjects_dir=subjects_dir)
    # mne.pick_info(mri.info, mne.pick_channels_regexp(mri.ch_names, "LTG..*"),
    #              copy=False)
    epoch_length = 2  # seconds
    events, event_id = mne.events_from_annotations(mri)
    epochs = mne.Epochs(mri, events, event_id=event_id['Audio'],
                        tmin=-0.5, tmax=-0.5 + epoch_length,
                        baseline=(None, 0), reject={'ecog': 1000e-6})
    # Make evoked from the one epoch and resample
    evoked = epochs.average()
    gamma_power_t = evoked.copy().filter(30, 90).apply_hilbert(envelope=True)
    gamma_info = gamma_power_t.info

    xyz_pts = np.array([dig['r'] for dig in evoked.info['dig']])
    gui = mne.gui.locate_ieeg(mri.info, subj_trans, CT_aligned,
                              subject=subject,
                              subjects_dir=subjects_dir, verbose=10)
    src = mne.read_source_spaces(
        op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    stc = mne.stc_near_sensors(gamma_power_t, trans='fsaverage',
                               subject='fsaverage', src=src,
                               mode='nearest', subjects_dir=subjects_dir,
                               distance=0.02)
    vmin, vmid, vmax = np.percentile(gamma_power_t.data, [10, 25, 90])
    clim = dict(kind='value', lims=[vmin, vmid, vmax])
    brain = stc.plot(surface='pial', hemi='rh', colormap='inferno',
                     colorbar=False,
                     clim=clim, views=['lat', 'med'],
                     subjects_dir=subjects_dir,
                     size=(250, 250), smoothing_steps='nearest',
                     time_viewer=False)
    brain.add_sensors(evoked.info, trans='fsaverage')
    """
    # mne.viz.plot_alignment(mri.info, subj_trans, subject,
    #                        subjects_dir=subjects_dir, ecog=True,
    #                        surfaces=['pial'], coord_frame='mri')
    # show_brain(filt, subj_trans, subject, subjects_dir)
    #D_dat_raw, D_dat_filt = find_dat(op.join(LAB_root, "D_Data",
    #                                 TASK, SUB))
    #raw_dat = open_dat_file(D_dat_raw, raw.copy().channels)
    #dat = open_dat_file(D_dat_filt, raw.copy().channels)
    # raw = raw_from_layout(layout, sub_pad, 1)
    # data = [raw, filt]
    # figure_compare(data, [ "BIDS Un", "BIDS "])
