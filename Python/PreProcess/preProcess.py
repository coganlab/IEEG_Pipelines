from bids import BIDSLayout
from bids.layout import BIDSFile
import mne
import os
from joblib import cpu_count
from mne_bids import read_raw_bids, BIDSPath
import numpy as np
from typing import Union, List, Tuple, Dict

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
BIDS_root = os.path.join(LAB_root, "BIDS-1.1_Uniqueness_point", "BIDS")
D59_dat = os.path.join(LAB_root, "D_Data", "Uniqueness_Point", "D59", "210720",
                       "001", "D59_Uniqueness_Point_210720.ieeg.dat")


def line_filter(data: mne.io.Raw) -> mne.io.Raw:
    if not data.preload:
        data.load_data()
    filt = data.copy().notch_filter(freqs=(60, 120, 180, 240),
                                    # method='fir',
                                    # fir_window='blackman',
                                    # pad='reflect',
                                    method='spectrum_fit',
                                    mt_bandwidth=20.0,
                                    filter_length='20s',
                                    p_value=0.01,
                                    verbose=10,
                                    n_jobs=cpu_count()-1)
    # make njobs 'cuda' with a gpu
    return filt


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
    runs = layout.get(return_type="id", target="run", subject=subject, run=run)
    raw = []
    for r in runs:
        BIDS_path = bidspath_from_layout(layout, subject=subject, run=r,
                                         extension=".edf")
        raw.append(read_raw_bids(bids_path=BIDS_path))
        raw[-1].load_data()
    whole_raw = mne.concatenate_raws(raw)
    return whole_raw


def open_dat_file(file_path: str, channels: Union[str, List[str], int],
                  sfreq: int = 2048, types: str = "ecog") -> mne.io.RawArray:
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
              ) -> Tuple[Dict[str, Dict[int, mne.io.Raw]]]:
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


def figure_compare(raw: List[mne.io.Raw], labels: List[str]):
    for title, data in zip(labels, raw):
        fig = data.plot_psd(fmax=250, average=True)
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large',
                     weight='bold')
        add_arrows(fig.axes[:2])


if __name__ == "__main__":
    layout = BIDSLayout(BIDS_root)
    raw = raw_from_layout(layout, "D0059", [1, 2, 3, 4])
    raw_dat = open_dat_file(D59_dat, raw.copy().ch_names)
    #raw.load_data()
    #filt = line_filter(raw)
    # raw_dat, dat = filt_main(layout, "D0028", 1)
    #data = [raw.copy(), filt.copy(), raw_dat.copy()]
    #figure_compare(data, ['Un', 'Spec_fit ', 'Correct '])

