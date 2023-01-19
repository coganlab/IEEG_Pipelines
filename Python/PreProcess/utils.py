import operator
import os.path as op
import pandas as pd
from os import PathLike as PL
from os import environ
from typing import List, TypeVar, Iterable, Union

import matplotlib as mpl
from matplotlib.pyplot import Figure, Axes
from mne.io import Raw
from mne.utils import config
import numpy as np
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

mpl.use("TkAgg")
HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
PathLike = TypeVar("PathLike", str, PL)


# plotting funcs

def figure_compare(raw: List[Raw], labels: List[str], avg: bool = True):
    """Plots the psd of a list of raw objects"""
    for title, data in zip(labels, raw):
        title: str
        data: Raw
        fig: Figure = data.plot_psd(fmax=250, average=avg, n_jobs=cpu_count(),
                                    spatial_colors=False)
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large',
                     weight='bold')
        add_arrows(fig.axes[:2])


def add_arrows(axes: Axes):
    """add some arrows at 60 Hz and its harmonics"""
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)


def ensure_int(x, name='unknown', must_be='an int', *, extra=''):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    extra = f' {extra}' if extra else extra
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError(f'{name} must be {must_be}{extra}, got {type(x)}')
    return x


def validate_type(item, types):
    try:
        if isinstance(types, TypeVar):
            check = isinstance(item, types.__constraints__)
        elif types is int:
            ensure_int(item)
            check = True
        elif types is float:
            check = is_number(item)
        else:
            check = isinstance(item, types)
    except TypeError:
        check = False
    if not check:
        raise TypeError(
            f"must be an instance of {types}, "
            f"got {type(item)} instead.")


def is_number(s) -> bool:
    if isinstance(s, str):
        try:
            float(s)
            return True
        except ValueError:
            return False
    elif isinstance(s, (np.number, int, float)):
        return True
    elif isinstance(s, pd.DataFrame):
        try:
            s.astype(float)
            return True
        except Exception:
            return False
    elif isinstance(s, pd.Series):
        try:
            pd.to_numeric(s)
            return True
        except Exception:
            return False
    else:
        return False


def parallelize(func: object, par_var: Iterable, n_jobs: int = None, *args,
                **kwargs) -> list:
    if n_jobs is None:
        n_jobs = cpu_count()
    settings = dict(verbose=5,  # prefer='threads',
                    pre_dispatch=n_jobs)
    env = dict(**environ)
    if config.get_config('MNE_CACHE_DIR') is not None:
        settings['temp_folder'] = config.get_config('MNE_CACHE_DIR')
    elif 'TEMP' in env.keys():
        settings['temp_folder'] = env['TEMP']

    if config.get_config('MNE_MEMMAP_MIN_SIZE') is not None:
        settings['max_nbytes'] = config.get_config('MNE_MEMMAP_MIN_SIZE')
    else:
        settings['max_nbytes'] = get_mem()

    data_new = Parallel(n_jobs, **settings)(delayed(func)(
        x_, *args, **kwargs)for x_ in tqdm(par_var))
    return data_new


def get_mem() -> Union[float, int]:
    from psutil import virtual_memory
    ram_per = virtual_memory()[3]/cpu_count()
    return ram_per
