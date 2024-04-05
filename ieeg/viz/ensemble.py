import gc
from functools import partial

import numpy as np
from joblib import cpu_count
from matplotlib import gridspec
from mne.io import Raw

from ieeg import Doubles, Signal
from ieeg.calc import stats
from ieeg.viz import _qt_backend

_qt_backend()

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
        _add_arrows(fig.axes[:2])
        fig.show()
        gc.collect()


def _add_arrows(axes: plt.Axes):
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
              size: tuple[int, int] = (12, 18), show: bool = True, **kwargs
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


def plot_dist(mat: iter, axis: int = 0, mode: str = 'sem',
              mask: np.ndarray = None, times: Doubles = None,
              label: str | int | float = None, color: str | list[int] = None,
              ax: plt.Axes = None) -> plt.Axes:
    """Plot the distribution for a single signal

    A distribution is the mean of the signal over the last dimension, with
    optional masking

    Parameters
    ----------
    mat : iter
        The signal to plot
    axis : int, optional
        The axis to use for the distribution, by default 0
    mode : str, optional
        The mode to use for the distribution, by default 'sem'
    mask : np.ndarray, optional
        The mask to use for the distribution, by default None
    times : Doubles, optional
        The times to use for the x-axis, by default None
    label : Union[str, int, float], optional
        The label for the signal, by default None
    color : Union[str, list[int]], optional
        The color to use for the signal, by default None
    ax : plt.Axes, optional
        The axes to plot on, by default None

    Returns
    -------
    plt.Axes
        The axes containing the plot
        """
    mean, std = stats.dist(mat, axis=axis, where=mask, mode=mode)
    if times is None:
        tscale = range(len(mean))
    else:
        tscale = np.linspace(times[0], times[1], len(mean))
    if ax is None:
        plt.figure()
        ax = plt.gca()
    p = ax.plot(tscale, mean, label=label, color=color)
    if color is None:
        color = p[-1].get_color()
    ax.fill_between(tscale, mean - std, mean + std, alpha=0.2, color=color)
    return ax


def plot_weight_dist(data: np.ndarray, label: np.ndarray, mode: str = 'sem',
                     mask: np.ndarray = None, times: Doubles = None,
                     sig_titles: list[str] = None,
                     colors: list[str | list[int]] = None, ax=None
                     ) -> (plt.Figure, plt.Axes):
    """Basic distribution plot for weighted signals

    Parameters
    ----------
    data : np.ndarray
        The data to plot
    label : np.ndarray
        The labels for the data
    mode : str, optional
        The mode to use for the distribution, by default 'sem'
    mask : np.ndarray, optional
        The mask to use for the distribution, by default None
    times : Doubles, optional
        The times to use for the x-axis
    sig_titles : list[str], optional
        The titles for the signals, by default None
    colors : list[str | list[int]], optional
        The colors for the signals, by default None
    ax : plt.Axes, optional
        The axes to plot on, by default None

    Returns
    -------
    plt.Figure
        The figure containing the plot
    plt.Axes
        The axes containing the plot
    """
    if ax is None:
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
            # w_sigs = np.average(data, weights=label[:, i], axis=0)
        ax = plot_dist(w_sigs, 0, mode, mask, times, label=stitle,
                       ax=ax, color=color)

    return ax


def subgrids(rows: int, cols: int, sub_cols: int,
             major_rows: tuple[int, ...] = (), titles: list = "",
             ylabels: list = "", xlabels: list = "", **kwargs
             ) -> (plt.Figure, plt.Axes):
    """Create a figure with subgrids

    Parameters
    ----------
    rows : int
        The number of rows in the figure
    cols : int
        The number of columns in the figure
    sub_cols : int
        The number of columns in the subgrids
    major_rows : tuple[int, ...], optional
        The rows that have major subgrids, by default ()
    titles : list, optional
        The titles for the subgrids, by default ""
    ylabels : list, optional
        The ylabels for the subgrids, by default ""
    xlabels : list, optional
        The xlabels for the subgrids, by default ""
    kwargs : dict
        Additional keyword arguments to pass to gridspec

    Returns
    -------
    (plt.Figure, plt.Axes)
        The figure and axes containing the subgrids
    """
    fig = plt.figure()
    gs = fig.add_gridspec(rows, cols, **kwargs)

    labels = dict(title=titles, ylabel=ylabels, xlabel=xlabels)
    for ltype, llist in labels.items():
        if isinstance(llist, str):
            if ltype in ("title", "xlabel"):
                labels[ltype] = [llist] * cols
            else:
                labels[ltype] = [llist] * rows
        elif len(llist) != cols and ltype in ("title", "xlabel"):
            raise ValueError(f"Length of {ltype} must be equal to cols")
        elif len(llist) != rows and ltype in ("ylabel",):
            raise ValueError(f"Length of {ltype} must be equal to rows")

    # Create subplots
    axs = [[None] * cols for _ in range(rows)]
    for r in range(rows):  # Only for the first two rows
        if r in major_rows:
            tc = 1
        else:
            tc = sub_cols
        for c in range(cols):
            gs0 = gs[r, c].subgridspec(1, tc, wspace=0, hspace=0)
            axs[r][c] = gs0.subplots(sharey=True, subplot_kw=dict(frameon=True)
                                     )

            # axes labels
            if r == 0:
                gs0.figure.suptitle(labels["title"][c])
            elif r == rows - 1:
                gs0.figure.supxlabel(labels["xlabel"][c])
            if c == 0:
                gs0.figure.supylabel(labels["ylabel"][r])

    return gs.figure, axs


if __name__ == "__main__":

    gs = subgrids(3, 3, 3, major_rows=(0, 1))
