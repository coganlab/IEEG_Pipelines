import re
from functools import singledispatch
from os import scandir, mkdir, path as op, walk
import pandas as pd
from bids.layout import BIDSFile, parse_file_entities
from mne.utils import fill_doc, verbose
from mne_bids import BIDSPath, get_bids_path_from_fname, mark_channels, \
    read_raw_bids, write_raw_bids
from mne_bids.write import _from_tsv
import os
from collections import OrderedDict
import mne
import numpy as np
from bids import BIDSLayout
from tqdm import tqdm

from ieeg import Doubles, PathLike
from joblib import Parallel, delayed
from ieeg import Signal
from itertools import product

mne.set_log_level("ERROR")
tfr_types = (mne.time_frequency.EpochsTFR, mne.time_frequency.AverageTFR)

class DataLoader:
    def __init__(self, layout: BIDSLayout, conds: dict[str, Doubles],
                 value_type: str = "zscore", avg: bool = True,
                 derivatives_folder: PathLike = 'stats', ext: str = '.fif'):
        self.root = layout.root
        self.subjects = sorted(layout.get_subjects())
        self.conds = conds
        self.value_type = value_type
        self.avg = avg
        self.derivatives_folder = derivatives_folder
        self.reader, self.suffix = self._get_reader_and_suffix(ext)

    def _get_reader_and_suffix(self, ext):
        allowed = ["zscore", "power", "significance", "pval"]
        assert ext in ('.fif', '.h5'), "ext must be one of ('.fif', '.h5')"

        match self.value_type:
            case "zscore":
                suffix = "zscore"
                if ext == ".fif":
                    suffix += "-epo" + ext
                    reader = lambda f: mne.read_epochs(f, False, preload=True)
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case "power":
                suffix = "power"
                if ext == ".fif":
                    suffix += "-epo" + ext
                    reader = lambda f: mne.read_epochs(f, False, preload=True)
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case "significance":
                suffix = "mask"
                if ext == ".fif":
                    suffix += "-ave" + ext
                    reader = mne.read_evokeds
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case "pval":
                suffix = "pval"
                if ext == ".fif":
                    suffix += "-ave" + ext
                    reader = mne.read_evokeds
                else:
                    suffix += "-tfr" + ext
                    reader = mne.time_frequency.read_tfrs
            case _:
                raise ValueError(f"value_type must be one of {allowed},"
                                 f" instead got {self.value_type}")
        return reader, suffix

    def load_subject_condition(self, subject, cond, dtype=None):
        out_cond = OrderedDict()
        try:
            fname = os.path.join(self.root, 'derivatives',
                                 self.derivatives_folder,
                                 f"{subject}_{cond}_{self.suffix}")
            epoch = self.reader(fname)
        except (FileNotFoundError, OSError) as e:
            mne.utils.logger.warn(e)
            return subject, cond, None

        sig = epoch
        times = self.conds[cond]
        if (self.suffix.split('.')[0].endswith("epo") or
                isinstance(sig, mne.time_frequency.EpochsTFR)):
            if self.avg:
                sig = sig.average(method=lambda x: np.nanmean(x, axis=0))
        elif isinstance(sig, list):
            sig = sig[0]
        mat = get_data(sig, tmin=times[0], tmax=times[1])
        if dtype is not None:
            mat = mat.astype(dtype)

        for i, ch in enumerate(sig.ch_names):
            if (self.suffix.split('.')[0].endswith("epo") or
                    isinstance(sig, mne.time_frequency.EpochsTFR)):
                for ev, id in sig.event_id.items():
                    ev = ev.split('/')[-1]
                    out_cond.setdefault(ev, {}).setdefault(ch, {})
                    if isinstance(sig, mne.time_frequency.EpochsTFR):
                        for j, f in enumerate(sig.freqs):
                            out_cond[ev][ch].setdefault(f, {})
                            out_cond[ev][ch][f] = mat[sig.events[:, 2] == id, i, j]
                    else:
                        out_cond[ev][ch] = mat[sig.events[:, 2] == id, i]
            elif isinstance(sig, mne.time_frequency.AverageTFR):
                for j, f in enumerate(sig.freqs):
                    out_cond.setdefault(ch, {}).setdefault(f, {})
                    out_cond[ch][f] = mat[i, j]
            else:
                out_cond[ch] = mat[i]
        return subject, cond, out_cond

    def load_dict(self, dtype=None, **kwargs):
        out = OrderedDict()
        combos = product(self.subjects, self.conds.keys())

        # joblib settings with some defaults
        kwargs.setdefault("n_jobs", 1) # cpu_count())
        kwargs.setdefault("return_as", "generator")
        kwargs.setdefault("backend", "loky")
        kwargs.setdefault("verbose", 0)

        proc = Parallel(**kwargs)(delayed(self.load_subject_condition)(
            subject, cond, dtype) for subject, cond in combos)
        for subject, cond, result in tqdm(
                proc,
                total=len(self.subjects) * len(self.conds),
                desc=f"(n_jobs={kwargs['n_jobs']}) Loading {self.value_type}",
                unit="files"):
            if result is not None:
                out.setdefault(subject, OrderedDict())[cond] = result

        return out


def get_data(inst: Signal, tmin: float, tmax: float):

    if isinstance(inst, tfr_types) or getattr(inst, "preload", False):
        tmin_idx = np.searchsorted(inst.times, tmin)
        tmax_idx = np.searchsorted(inst.times, tmax, side="right")
        return inst._data[..., tmin_idx:tmax_idx]
    else:
        return inst.get_data(tmin=tmin, tmax=tmax)
def find_dat(folder: PathLike) -> (PathLike, PathLike):
    """Looks for the .dat file in a specified folder

    Parameters
    ----------
    folder : PathLike
        The folder to search in.

    Returns
    -------
    Tuple[PathLike, PathLike]
        The paths to the ieeg and cleanieeg files.
    """
    cleanieeg = None
    ieeg = None
    for root, _, files in walk(folder):
        for file in files:
            if re.match(r".*cleanieeg\.dat.*", file):
                cleanieeg: PathLike = op.join(root, file)
            elif re.match(r".*ieeg\.dat.*", file):
                ieeg: PathLike = op.join(root, file)
            if ieeg is not None and cleanieeg is not None:
                return ieeg, cleanieeg
    raise FileNotFoundError("Not all .dat files were found:")

def dict_to_structured_array(dict_matrices, filename='structured_array.npy'):
    """Save a dictionary of matrices to a structured array."""
    # Get the keys and shapes
    keys = list(dict_matrices.keys())
    shape = dict_matrices[keys[0]].shape

    # Create a data type for the structured array
    dt = np.dtype([(key, dict_matrices[key].dtype, shape) for key in keys])

    # Create the structured array
    structured_array = np.zeros((1,), dtype=dt)

    # Fill the structured array
    for key in keys:
        structured_array[key] = dict_matrices[key]

    # Save the structured array to a file
    np.save(filename, structured_array)

def bidspath_from_layout(layout: BIDSLayout, **kwargs) -> BIDSPath:
    """Searches a :class:`BIDSLayout` for a file and returns a
    :class:`BIDSPath` to it.

    Parameters
    ----------
    layout : BIDSLayout
        The :class:`BIDSLayout` to search.
    **kwargs : dict
        The parameters passed to :func:`BIDSLayout.get`

    Returns
    -------
    BIDS_path: BIDSPath
        The BIDSPath to the file.
    """
    my_search: list[BIDSFile] = layout.get(**kwargs)
    if len(my_search) >= 2:
        raise FileNotFoundError("Search terms matched more than one file: \n"
                                "{} \n try adding more search terms"
                                "".format(my_search))
    elif len(my_search) == 0:
        raise FileNotFoundError("No files match your search terms")
    found = my_search[0]
    entities = found.get_entities()
    if 'desc' in entities:
        entities['description'] = entities.pop('desc')
    BIDS_path = BIDSPath(root=layout.root, **entities)
    return BIDS_path


@fill_doc
def raw_from_layout(layout: BIDSLayout, preload: bool = True,
                    run: list[int] | int = None, **kwargs) -> mne.io.Raw:
    """Searches a BIDSLayout for a raw file and returns a mne Raw object.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDSLayout to search.
    %(preload)s
    run : Union[List[int], int], optional
        The run to search for, by default None
    **kwargs : dict
        The parameters passed to bids.BIDSLayout.get()

    Returns
    -------
    mne.io.Raw

    Examples
    --------

    >>> import mne
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=False)
    >>> layout = BIDSLayout(bids_root)
    >>> raw = raw_from_layout(layout, subject="pt1", preload=True,
    ... extension=".vhdr", verbose=False)
    Reading 0 ... 269079  =      0.000 ...   269.079 secs...
    """
    verbose = kwargs.pop('verbose', True)
    if run is None:
        runs = layout.get(return_type="id", target="run", **kwargs)
    else:
        runs = list(run)
    raw: list[mne.io.Raw] = []
    if runs:
        for r in runs:
            BIDS_path = bidspath_from_layout(layout, run=r, **kwargs)
            new_raw = read_raw_bids(bids_path=BIDS_path, verbose=verbose)
            raw.append(new_raw.copy())
        whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    else:
        BIDS_path = bidspath_from_layout(layout, **kwargs)
        whole_raw = read_raw_bids(bids_path=BIDS_path, verbose=verbose)
    if preload:
        whole_raw.load_data()
    return whole_raw


def open_dat_file(file_path: str, channels: list[str], sfreq: int = 2048,
                  types: str = "seeg", units: str = "uV") -> mne.io.RawArray:
    """Opens a .dat file and returns a mne.io.RawArray object.

    Parameters
    ----------
    file_path : str
        The path to the .dat file.
    channels : List[str]
        The channels to load.
    sfreq : int, optional
        The sampling frequency, by default 2048
    types : str, optional
        The channel types, by default "seeg"
    units : str, optional
        The units of the data, by default "uV"

    Returns
    -------
    mne.io.RawArray
    """
    with open(file_path, mode='rb') as f:
        data = np.fromfile(f, dtype="float32")
    if "Trigger" in channels:
        channels.remove("Trigger")
    array = np.reshape(data, [len(channels), -1], order='F')
    match units:
        case "V":
            factor = 1
        case "mV":
            factor = 1e-3
        case "uV":
            factor = 1e-6
        case "nV":
            factor = 1e-9
        case _:
            raise NotImplementedError("Unit " + units + " not implemented yet")
    info = mne.create_info(channels, sfreq, types)
    raw = mne.io.RawArray(array * factor, info)
    return raw


def get_data(task: str, root: PathLike,
             prefix: str = r"BIDS-\d\.\d_") -> BIDSLayout:
    """Gets the data for a subject and task.

    Parameters
    ----------
    task : str
        The task to get the data for, by default "SentenceRep"
    root : PathLike
        The path to the lab directory, by default LAB_root
    prefix : str, optional
        The prefix of the BIDS directory, by default 'BIDS'

    Returns
    -------
    layout : BIDSLayout
        The BIDSLayout for the subject.

    Examples
    --------

    >>> import mne
    >>> bids_root = mne.datasets.epilepsy_ecog.data_path(verbose=False)
    >>> parent = op.dirname(bids_root)
    >>> get_data('epilepsy-ecog-data', parent, "MNE-") # doctest: +ELLIPSIS
    BIDS Layout: ... | Subjects: 1 | Sessions: 1 | Runs: 0
    """
    # scan data directory
    scan = scandir(root)

    # keep only matching BIDS directories
    matches = filter(lambda x: re.match(prefix + task, x.name), scan)

    # check that there is at least one match
    ordered = sorted(matches, key=lambda x: x.name)
    assert len(ordered) > 0, FileNotFoundError(
        "Could not find BIDS directory in {} for task {}".format(root, task))

    # grab the last match
    BIDS_root = op.join(root, ordered[-1].name)

    # check for BIDS subfolder
    if op.isdir(alt_root := op.join(BIDS_root, "BIDS")):
        return BIDSLayout(alt_root, derivatives=True)
    else:
        return BIDSLayout(BIDS_root, derivatives=True)


@fill_doc
@verbose
def save_derivative(inst: Signal, layout: BIDSLayout, pipeline: str = None,
                    overwrite: bool = False, format: str = 'EDF',
                    anonymize: bool = True, verbose=None):
    """Save an intermediate data instance from a pipeline to a BIDS folder.

    Parameters
    ----------
    inst : Signal
        The data instance to save.
    layout : BIDSLayout
        The BIDSLayout of the original data.
    pipeline : str
        The name of the pipeline.
    %(overwrite)s
    format : str
        The format to save the data in. Defaults to EDF.
    %(verbose)s
    """
    save_dir = op.join(layout.root, "derivatives", pipeline)
    if not op.isdir(save_dir):
        mkdir(save_dir)
    bounds = inst.annotations.copy()
    starttime = inst.first_time
    endtime = inst.times[-1]+inst.first_time
    bounds = bounds[np.where(bounds.description == 'BAD boundary')[0]]
    bounds = [starttime] + list(bounds.onset) + [endtime]
    for i, file in enumerate(inst.filenames):
        entities = parse_file_entities(file)
        if 'desc' in entities.keys():
            entities['description'] = entities.pop('desc')
        if 'subject' not in entities.keys():
            sub = inst.info['subject_info']['his_id']
            entities['subject'] = sub[slice(4, None)]
        if 'task' not in entities.keys():
            entities['task'] = ''
        if pipeline:
            entities['description'] = pipeline
        bids_path = BIDSPath(**entities, root=save_dir)

        # account for cropping
        run = inst.copy().crop(tmin=bounds[i] - inst.first_time,
                               tmax=bounds[i + 1] - inst.first_time)
        if anonymize:
            if isinstance(run, Signal):
                run.anonymize()
                anonymize = None
            else:
                anonymize = {'daysback': 10000000}
        else:
            anonymize = None

        write_raw_bids(run, bids_path, allow_preload=True, format=format,
                       acpc_aligned=True, overwrite=overwrite,
                       anonymize=anonymize, verbose=verbose)


def get_bad_chans(fname: str):
    """Gets the bad channels corresponding to a file.

    Parameters
    ----------
    fname : str
        The path to the file.

    Returns
    -------
    bads : list[str]
        The bad channels.
    """
    data = _from_tsv(fname.replace("_ieeg.edf", "_channels.tsv"))
    bads = [n for n, s in zip(data['name'], data['status']) if s == 'bad']
    return bads


@singledispatch
@verbose
def update(filename: PathLike, channels: list[str],
           description: list[str] | str = None, status: str = 'good',
           verbose=None):
    """Updates the files of a data instance with current metadata

    Parameters
    ----------
    filename : PathLike
        The path to the file to update.
    channels : list[str]
        The channels to update.
    description : list[str] | str, optional
        The description of the channels, by default None
    status : str, optional
        The status of the channels, by default 'good'
    """
    if isinstance(description, str):
        description = [description for _ in range(len(channels))]

    bids_path = get_bids_path_from_fname(filename)
    mark_channels(bids_path, ch_names=channels, status=status,
                  descriptions=description, verbose=verbose)


@update.register
def _(inst: mne.io.base.BaseRaw,
      layout: BIDSLayout, description: list[str] | str = None, verbose=None):
    if not hasattr(inst, 'filenames'):
        inst.filenames = inst.info['subject_info'].get('files', None)
    for i, file in enumerate(inst.filenames):
        fname = op.join(layout.root, file)
        update(fname, inst.info['bads'], description=description, status='bad',
               verbose=verbose)
        goods = [ch for ch in inst.ch_names if ch not in inst.info['bads']]
        update(fname, channels=goods, status='good', verbose=None)


@update.register
def _(inst: mne.time_frequency.BaseTFR,
      layout: BIDSLayout, description: list[str] | str = None, verbose=None):
    if not hasattr(inst, 'filenames'):
        inst.filenames = inst.info['subject_info'].get('files', None)
    for i, file in enumerate(inst.filenames):
        fname = op.join(layout.root, file)
        update(fname, inst.info['bads'], description=description, status='bad',
               verbose=verbose)
        goods = [ch for ch in inst.ch_names if ch not in inst.info['bads']]
        update(fname, channels=goods, status='good', verbose=None)


def get_elec_volume_labels(subj: str, subj_dir: str, radius: int = 10,
                           atlas: str = ".a2009s") -> pd.DataFrame:
    """Gets the electrode volume labels for a subject.

    Parameters
    ----------
    subj : str
        The subject to get the labels for.
    subj_dir : str
        The directory of the subject.
    radius : int, optional
        The radius of the volume, by default 10
    atlas : str, optional
        The atlas to use, by default ".a2009s"

    Returns
    -------
    pd.DataFrame
        The labels of the electrode volume.
    """
    filename = op.join(subj_dir, subj, "elec_recon",
                       f"{subj}_elec_location_radius_{radius}mm_aparc{atlas}+"
                       f"aseg.mgz")
    if op.exists(filename + "_brainshifted.csv"):
        filename += "_brainshifted.csv"
    else:
        filename += ".csv"
    out = pd.read_csv(filename, header=None, skiprows=1, index_col=1)
    return out
