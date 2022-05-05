import os.path as op
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
    for title, data in zip(labels, raw):
        title: str
        data: Raw
        fig: Figure = data.plot_psd(fmax=250, average=avg, n_jobs=cpu_count())
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large',
                     weight='bold')
        add_arrows(fig.axes[:2])


def add_arrows(axes: Axes):
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
