import gc
from typing import List

import matplotlib as mpl
import numpy as np
from joblib import cpu_count
from matplotlib.pyplot import Axes
from mne.io import Raw

try:
    mpl.use("TkAgg")
except ImportError:
    pass


def figure_compare(raw: List[Raw], labels: List[str], avg: bool = True,
                   n_jobs: int = None, **kwargs):
    """Plots the psd of a list of raw objects"""
    if n_jobs is None:
        n_jobs = cpu_count() - 2
    for title, data in zip(labels, raw):
        title: str
        data: Raw
        psd = data.compute_psd(n_jobs=n_jobs, **kwargs,
                               n_fft=int(data.info['sfreq']))
        fig = psd.plot(average=avg, spatial_colors=avg)
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large',
                     weight='bold')
        add_arrows(fig.axes[:2])
        gc.collect()


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
