import operator
import os.path as op
import pandas as pd
from os import PathLike as PL
from typing import List, TypeVar

from matplotlib.pyplot import Figure, Axes
from mne.io import Raw
import numpy as np
from joblib import cpu_count


HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
PathLike = TypeVar("PathLike", str, PL)


# plotting funcs

def figure_compare(raw: List[Raw], labels: List[str], avg: bool = True):
    """Plots the psd of a list of raw objects"""
    for title, data in zip(labels, raw):
        title: str
        data: Raw
        fig: Figure = data.plot_psd(fmax=250, average=avg, n_jobs=cpu_count(), spatial_colors=False)
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


def sum_squared(X: np.ndarray) -> float:
    """Compute norm of an array.
    Parameters
    ----------
    X : array
        Data whose norm must be found.
    Returns
    -------
    value : float
        Sum of squares of the input array X.
    """
    X_flat = X.ravel(order='F' if np.isfortran(X) else 'C')
    return np.dot(X_flat, X_flat)
