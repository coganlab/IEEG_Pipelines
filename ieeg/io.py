from ieeg import PathLike, Signal
import re
from os import walk, listdir, mkdir, path as op
from functools import singledispatch

from bids.layout import BIDSFile, parse_file_entities
from mne_bids import read_raw_bids, BIDSPath, write_raw_bids, \
    mark_channels, get_bids_path_from_fname
from mne.utils import verbose, fill_doc
from mne_bids.write import _from_tsv
from bids import BIDSLayout
import numpy as np
import mne


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
    """
    if run is None:
        runs = layout.get(return_type="id", target="run", **kwargs)
    else:
        runs = list(run)
    raw: list[mne.io.Raw] = []
    if runs:
        for r in runs:
            BIDS_path = bidspath_from_layout(layout, run=r, **kwargs)
            new_raw = read_raw_bids(bids_path=BIDS_path)
            raw.append(new_raw.copy())
        whole_raw: mne.io.Raw = mne.concatenate_raws(raw)
    else:
        BIDS_path = bidspath_from_layout(layout, **kwargs)
        whole_raw = read_raw_bids(bids_path=BIDS_path)
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


def get_data(task: str, root: PathLike) -> BIDSLayout:
    """Gets the data for a subject and task.

    Parameters
    ----------
    task : str, optional
        The task to get the data for, by default "SentenceRep"
    root : PathLike, optional
        The path to the lab directory, by default LAB_root

    Returns
    -------
    layout : BIDSLayout
        The BIDSLayout for the subject.
    """
    BIDS_root = None
    for dir in listdir(root):
        if re.match(r"BIDS-\d\.\d_" + task, dir) and "BIDS" in listdir(op.join(
                root, dir)):
            BIDS_root = op.join(root, dir, "BIDS")
            break
    if BIDS_root is None:
        raise FileNotFoundError("Could not find BIDS directory in {} for task "
                                "{}".format(root, task))
    layout = BIDSLayout(BIDS_root, derivatives=True)
    return layout


@fill_doc
@verbose
def save_derivative(inst: Signal, layout: BIDSLayout, pipeline: str = None,
                    overwrite: bool = False, format: str = 'EDF',
                    verbose=None):
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
    bounds = bounds[np.where(bounds.description == 'BAD boundary')[0]]
    bounds = [0] + list(bounds.onset) + [inst.times[-1]]
    for i, file in enumerate(inst.filenames):
        entities = parse_file_entities(file)
        if 'desc' in entities.keys():
            entities['description'] = entities.pop('desc')
        if pipeline:
            entities['description'] = pipeline
        bids_path = BIDSPath(**entities, root=save_dir)
        run = inst.copy().crop(tmin=bounds[i], tmax=bounds[i+1])
        write_raw_bids(run, bids_path, allow_preload=True, format=format,
                       acpc_aligned=True, overwrite=overwrite, verbose=verbose)


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
def _(inst: mne.time_frequency._BaseTFR,
      layout: BIDSLayout, description: list[str] | str = None, verbose=None):
    if not hasattr(inst, 'filenames'):
        inst.filenames = inst.info['subject_info'].get('files', None)
    for i, file in enumerate(inst.filenames):
        fname = op.join(layout.root, file)
        update(fname, inst.info['bads'], description=description, status='bad',
               verbose=verbose)
        goods = [ch for ch in inst.ch_names if ch not in inst.info['bads']]
        update(fname, channels=goods, status='good', verbose=None)
