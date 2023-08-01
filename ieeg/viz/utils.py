import gc
from functools import partial
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from joblib import cpu_count
from mne.io import Raw

from ieeg import Doubles, Signal
from ieeg.calc import stats


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
        fig.show()
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


def _onclick_select(event, inst, axs):
    for a in axs:
        if event.inaxes == a:
            ch = a.get_title()
            if ch in inst.info['bads']:
                inst.info['bads'].remove(ch)
                print(f"Removing {ch} from bads")
            else:
                inst.info['bads'].append(ch)
                print(f"adding {ch} to bads")


def chan_grid(inst: Signal, n_cols: int = 10, n_rows: int = 6,
              plot_func: callable = None, picks: list[str | int] = None,
              size: tuple[int, int] = (8, 12), show: bool = True, **kwargs
              ) -> list[plt.Figure]:
    """Plot a grid of the channels of a Signal object

    Parameters
    ----------
    size
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
    size : tuple[int, int], optional
        The size of the figure, by default (8, 12)
    show : bool, optional
        Whether to show the figure, by default True

    Returns
    -------
    list[plt.Figure]
        The figures containing the grid
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
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, frameon=False,
                                figsize=size)

        select = partial(_onclick_select, inst=inst, axs=fig.axes)
        text_spec = dict(fontsize=12, weight="extra bold")

        for j, chan in enumerate(chans[i * per_fig:(i + 1) * per_fig]):
            if j + 1 % n_cols == 0 or i == len(chans) - 1:
                bar = True
            else:
                bar = False
            if "colorbar" in plot_func.__code__.co_varnames:
                kwargs["colorbar"] = bar
            ax = axs.flatten()[j]
            plot_func(picks=[chan], axes=ax, show=False, **kwargs)
            ax.set_title(chan, pad=0, **text_spec)
            ax.tick_params(axis='both', which='major', labelsize=7,
                           direction="in")
            ax.set_xlabel("")
            ax.set_ylabel("")
            gc.collect()
        fig.supxlabel("Time (s)", **text_spec)
        fig.supylabel("Frequency (Hz)", **text_spec)
        if i == numfigs - 1:
            while j + 1 < n_cols * n_rows:
                j += 1
                ax = axs.flatten()[j]
                ax.axis("off")
        fig.canvas.mpl_connect("button_press_event", select)
        fig.tight_layout()
        figs.append(fig)
        if show:
            figs[i].show()
    return figs


def plot_dist(mat: iter, mask: np.ndarray = None, times: Doubles = None,
              label: str | int | float = None,
              color: str | list[int] = None) -> plt.Axes:
    """Plot the distribution for a single signal

    A distribution is the mean of the signal over the last dimension, with
    optional masking

    Parameters
    ----------
    mat : iter
        The signal to plot
    mask : np.ndarray, optional
        The mask to use for the distribution, by default None
    times : Doubles, optional
        The times to use for the x-axis, by default None
    label : Union[str, int, float], optional
        The label for the signal, by default None
    color : Union[str, list[int]], optional
        The color to use for the signal, by default None

    Returns
    -------
    plt.Axes
        The axes containing the plot
        """
    # mean, std = np.mean(mat, axis=0), np.std(mat, axis=0)
    mean, std = stats.dist(mat, mask)
    if times is None:
        tscale = range(len(mean))
    else:
        tscale = np.linspace(times[0], times[1], len(mean))
    p = plt.plot(tscale, mean, label=label, color=color)
    if color is None:
        color = p[-1].get_color()
    plt.fill_between(tscale, mean - std, mean + std, alpha=0.2, color=color)
    return plt.gca()


def plot_weight_dist(data: np.ndarray, label: np.ndarray,
                     mask: np.ndarray = None, sig_titles: list[str] = None,
                     colors: list[str | list[int]] = None
                     ) -> (plt.Figure, plt.Axes):
    """Basic distribution plot for weighted signals

    Parameters
    ----------
    data : np.ndarray
        The data to plot
    label : np.ndarray
        The labels for the data
    mask : np.ndarray, optional
        The mask to use for the distribution, by default None
    sig_titles : list[str], optional
        The titles for the signals, by default None
    colors : list[str | list[int]], optional
        The colors for the signals, by default None

    Returns
    -------
    plt.Figure
        The figure containing the plot
    plt.Axes
        The axes containing the plot
    """
    fig, ax = plt.subplots()
    if len(label.shape) > 1:
        group = range(min(np.shape(label)))
        weighted = True
    else:
        group = np.unique(label)
        weighted = False
    if sig_titles is None:
        sig_titles = [sig_titles] * len(group)
    if colors is None:
        colors = [colors] * len(group)
    for i, stitle, color in zip(group, sig_titles, colors):
        if not weighted:
            w_sigs = data[label == i]
        else:
            w_sigs = np.multiply(data.T, label[:, i]).T
        ax = plot_dist(w_sigs, mask, label=stitle, color=color)
    return fig, ax


if __name__ == "__main__":
    import numpy as np

    with open("../spectra.npy", "rb") as f:
        spectra = np.load(f, allow_pickle=True)[0]
    # spectra2 = np.load("spectra.npy", allow_pickle=True,)['spectra']

    chan_grid(spectra, vmin=0.7, vmax=1.4)
