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
    """Plots the psd of a list of raw objects

    Parameters
    ----------
    raw : list[Raw]
        The raw objects to plot
    labels : list[str]
        The labels for the raw objects
    avg : bool, optional
        Whether to average the psd over channels, by default True
    n_jobs : int, optional
        The number of jobs to use for the computation, by default None
    **kwargs
        Additional keyword arguments to pass to Raw.compute_psd()
    """
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


def chan_grid(inst: Signal, n_cols: int = 10, n_rows: int = 6,
              plot_func: callable = None, picks: list[Union[str, int]] = None,
              **kwargs) -> list[plt.Figure]:
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

    per_fig = n_cols * n_rows
    numfigs = int(np.ceil(len(chans) / per_fig))
    figs = []
    for i in range(numfigs):
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, frameon=False)
        for j, chan in enumerate(chans[i * per_fig:(i + 1) * per_fig]):
            if j + 1 % n_cols == 0 or i == len(chans) - 1:
                bar = True
            else:
                bar = False
            if "colorbar" in plot_func.__code__.co_varnames:
                kwargs["colorbar"] = bar
            ax = axs.flatten()[j]
            plot_func(picks=[chan], axes=ax, show=False, **kwargs)
            ax.set_title(chan, fontsize=8, pad=0)
            ax.tick_params(axis='both', which='major', labelsize=6,
                           direction="in")
            ax.set_xlabel("")
            ax.set_ylabel("")
            gc.collect()
        fig.supxlabel("Time (s)")
        fig.supylabel("Frequency (Hz)")
        if i == numfigs - 1:
            while j + 1 < n_cols * n_rows:
                j += 1
                ax = axs.flatten()[j]
                ax.axis("off")
        figs.append(fig)
        figs[i].show()
    return figs


if __name__ == "__main__":
    import mne
    import numpy as np

    with open("../spectra.npy", "rb") as f:
        spectra = np.load(f, allow_pickle=True)[0]
    # spectra2 = np.load("spectra.npy", allow_pickle=True,)['spectra']
    from PreProcess.utils import plotting

    plotting.chan_grid(spectra, vmin=0.7, vmax=1.4)
