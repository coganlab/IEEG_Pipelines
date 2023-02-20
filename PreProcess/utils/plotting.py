import gc
from typing import Union

import matplotlib as mpl
import numpy as np
from joblib import cpu_count
from mne.io import Raw

from PreProcess.timefreq.utils import Signal

try:
    mpl.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt  # noqa: E402


def figure_compare(raw: list[Raw], labels: list[str], avg: bool = True,
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


def add_arrows(axes: plt.Axes):
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


def chan_grid(inst: Signal, n_cols: int = 10, n_rows: int = None,
              plot_func: callable = None, picks: list[Union[str, int]] = None,
              **kwargs) -> plt.Figure:
    """Plot a grid of the channels of a Signal object

    Parameters
    ----------
    inst : Signal
        The Signal object to plot
    n_cols : int, optional
        Number of columns in the grid, by default 10
    n_rows : int, optional
        Number of rows in the grid, by default the minimum number of rows
    plot_func : callable, optional
        The function to use to plot the channels, by default inst.plot()
    picks : list[Union[str, int]], optional
        The channels to plot, by default all

    Returns
    -------
    plt.Figure
        The figure containing the grid
    """
    if n_rows is None:
        n_rows = int(np.ceil(len(inst.ch_names) / n_cols))
    if plot_func is None:
        plot_func = inst.plot
    if picks is None:
        chans = inst.ch_names
    elif isinstance(picks[0], str):
        chans = picks
    elif isinstance(picks[0], int):
        chans = [inst.ch_names[i] for i in picks]
    else:
        raise TypeError("picks must be a list of str or int")

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, frameon=False)
    for i, chan in enumerate(chans):
        if i + 1 % n_cols == 0 or i == len(chans) - 1:
            bar = True
        else:
            bar = False
        if "colorbar" in plot_func.__code__.co_varnames:
            kwargs["colorbar"] = bar
        plot_func(picks=[chan], axes=axs[i], **kwargs)
        axs[i].set_title(chan)
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")

    while i + 1 < n_cols * n_rows:
        i += 1
        axs[i].axis("off")

    fig.supxlabel("Time (s)")
    fig.supylabel("Frequency (Hz)")
    return fig
