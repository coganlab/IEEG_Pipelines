import inspect
import operator
from itertools import chain
from os import environ
from typing import Generator, Iterable, TypeVar, Union

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from mne.utils import config, logger
from scipy.signal import get_window


def iterate_axes(arr: np.ndarray, axes: tuple[int, ...], index=(), axis=0):
    """Iterate over all possible indices for a set of axes

    Parameters
    ----------
    arr : np.ndarray
        The array to iterate over
    axes : tuple[int]
        The axes to iterate over
    index : tuple[int]
        The current index
    axis : int
        The current axis

    Yields
    ------
    tuple[slice]
        The indices for the current iteration

    Examples
    --------
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> for sl in iterate_axes(arr, (0, 1)):
    ...     print(arr[sl])
    [0 1 2 3]
    [4 5 6 7]
    [ 8  9 10 11]
    [12 13 14 15]
    [16 17 18 19]
    [20 21 22 23]
    """
    if axis < len(axes):
        for i in range(arr.shape[axes[axis]]):
            yield from iterate_axes(arr, axes, index + (i,), axis + 1)
    else:
        # Create a tuple of slices for all axes
        slices = [slice(None)] * arr.ndim
        for axis, i in zip(axes, index):
            slices[axis] = i
        yield tuple(slices)


def ensure_int(x, name: str = 'unknown', must_be: str = 'an int', *, extra=''):
    """Ensure a variable is an integer.

    Parameters
    ----------
    x : object
        The object to check.
    name : str
        The name of the variable to check.
    must_be : str
        The type of the variable to check.
    extra : str
        Extra text to add to the error message.

    Notes
    -----
    This is preferred over numbers.Integral, see:
    https://github.com/scipy/scipy/pull/7351#issuecomment-299713159

    Examples
    --------
    >>> ensure_int(1)
    1
    >>> ensure_int(1.0)
    Traceback (most recent call last):
    ...
    TypeError: unknown must be an int, got <class 'float'>
    >>> ensure_int('1')
    Traceback (most recent call last):
    ...
    TypeError: unknown must be an int, got <class 'str'>
    >>> ensure_int('1.0', extra='a string')
    Traceback (most recent call last):
    ...
    TypeError: unknown must be an int a string, got <class 'str'>
    """

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
    """Validate the type of an object.

    Parameters
    ----------
    item : object
        The object to check.
    types : type
        The type to check against.
    """
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
    """Check if an object is a number

    Parameters
    ----------
    s : object
        The object to check

    Returns
    -------
    bool
        True if the object is a number, False otherwise
    """
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


def proc_array(func: callable, arr_in: np.ndarray, axes: int | tuple[int] = 0,
               n_jobs: int = None, desc: str = "Slices", inplace: bool = True,
               **kwargs) -> np.ndarray:
    """Execute a function in parallel over slices of an array

    Parameters
    ----------
    func : callable
        The function to execute
    arr_in : np.ndarray
        The array to slice
    axes : int | tuple[int]
        The axes to slice over
    n_jobs : int
        The number of jobs to run in parallel
    desc : str
        The description to use for the progress bar
    inplace : bool
        Whether to modify the input array in place

    Returns
    -------
    np.ndarray
        The output of the function, same shape as the input array

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> proc_array(square, np.arange(10))
    array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])
    """

    if isinstance(axes, int):
        axes = (axes,)

    if inplace:
        arr_out = arr_in
    else:
        arr_out = arr_in.copy()

    # Get the cross-section indices and array input generator
    cross_sect_ind = list(np.ndindex(*[arr_in.shape[axis] for axis in axes]))
    array_gen = list(arr_in[indices] for indices in cross_sect_ind)

    gen = Parallel(n_jobs, return_as='generator', verbose=40)(
        delayed(func)(x_, **kwargs) for x_ in array_gen)

    # Create process pool and apply the function in parallel
    for out, ind in zip(gen, cross_sect_ind):
        arr_out[ind] = out

    return arr_out


def parallelize(func: callable, ins: Iterable, verbose: int = 10,
                n_jobs: int = None, **kwargs) -> list | None:
    """Parallelize a function to run on multiple processors.

    This function is a wrapper for joblib.Parallel. It will automatically
    determine the number of jobs to run in parallel based on the number of
    cores available on the system. It will also automatically set the
    temp_folder and max_nbytes parameters for joblib.Parallel based on the
    MNE_CACHE_DIR and MNE_MEMMAP_MIN_SIZE parameters in mne-python's
    configuration file.

    Notes
    -----
    If the elements of the par_var iterable are tuples, the function will be
    called with the tuple unpacked, setting each item in the tuple to be
    assigned to a separate argument. If the elements of the par_var iterable
    are not tuples, the function will be called with the element as the first
    argument.


    Parameters
    ----------
    func : callable
        The function to parallelize
    ins : Iterable
        The iterable to parallelize over
    n_jobs : int
        The number of jobs to run in parallel. If None, will use all
        available cores. If -1, will use all available cores.
    **kwargs
        Additional keyword arguments to pass to the function

    Returns
    -------
    list
        The output of the function for each element in par_var

    Examples
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> parallelize(square, [1, 2, 3])
    [1, 4, 9]
    """

    assert ins
    if n_jobs is None:
        if 'n_jobs' in inspect.getfullargspec(func).args:
            kwargs['n_jobs'] = -2
        else:
            n_jobs = -2
    settings = dict(verbose=verbose)
    settings['prefer'] = kwargs.pop('prefer', None)
    settings['backend'] = kwargs.pop('backend', None)
    settings['mmap_mode'] = kwargs.pop('mmap_mode', 'r')
    settings['require'] = kwargs.pop('require', None)

    env = dict(**environ)
    if config.get_config('MNE_CACHE_DIR') is not None:
        settings['temp_folder'] = config.get_config('MNE_CACHE_DIR')
    elif 'TEMP' in env.keys():
        settings['temp_folder'] = env['TEMP']
    else:
        settings['temp_folder'] = None

    if config.get_config('MNE_MEMMAP_MIN_SIZE') is not None:
        settings['max_nbytes'] = config.get_config('MNE_MEMMAP_MIN_SIZE')
    else:
        settings['max_nbytes'] = get_mem()

    for var in ins:
        if isinstance(var, tuple):
            x_is_tup = True
        elif isinstance(ins, Generator):
            x_is_tup = False
            ins = chain((var,), ins)
        else:
            x_is_tup = False
        break

    if x_is_tup:
        return Parallel(n_jobs, **settings)(delayed(func)(
            *x_, **kwargs) for x_ in ins)
    else:
        return Parallel(n_jobs, **settings)(delayed(func)(
            x_, **kwargs) for x_ in ins)


def get_mem() -> int:
    """Get the amount of memory to use for parallelization.

    Returns
    -------
    float | int
        The amount of memory to use for parallelization
    """
    from psutil import virtual_memory
    ram_per = virtual_memory().available >> 0 // cpu_count()
    return ram_per


def sliding_window(x_data: np.ndarray, labels: np.ndarray,
                   scorer: callable, window_size: int = 20, axis: int = -1,
                   n_jobs: int = -3, **kwargs) -> np.ndarray:
    """Compute a function over a sliding window.

    Parameters
    ----------
    x_data : np.ndarray, shape (..., trials, time)
        The data to compute the function over
    labels : np.ndarray, shape (trials,)
        The labels for each trial
    scorer : callable
        The function to compute over the sliding window. Must take two
        arguments, the data and the labels.
    window_size : int
        The size of the sliding window
    axis : int
        The axis to compute the sliding window over
    n_jobs : int
        The number of jobs to run in parallel

    Returns
    -------
    np.ndarray
        The output of the function, shape (..., time - window_size + 1)

    Examples
    --------
    >>> def square(x, labels):
    ...     return np.mean(x ** 2, where=labels == 1)
    >>> x_data = np.arange(40).reshape(4, 10)
    >>> labels = np.array([0, 1, 1])
    >>> sliding_window(x_data, labels, square, window_size=3)
    array([397.5, 431.5, 467.5, 505.5, 545.5, 587.5, 631.5])
    """

    # make windowing generator
    axis = x_data.ndim + axis if axis < 0 else axis
    slices = (slice(start, start + window_size)
              for start in range(0, x_data.shape[axis] - window_size))
    idxs = (tuple(slice(None) if i != axis else sl for i in
                  range(x_data.ndim)) for sl in slices)

    # Use joblib to parallelize the computation
    gen = Parallel(n_jobs=n_jobs, return_as='generator', verbose=40)(
        delayed(scorer)(x_data[idx], labels, **kwargs) for idx in idxs)

    # initialize output array by running 1 job and get the shape
    mat = next(gen)
    out = np.zeros((x_data.shape[axis] - window_size, *mat.shape),
                   dtype=mat.dtype)
    out[0] = mat

    # fill in the rest of the output array
    for i, mat in enumerate(gen):
        out[i + 1] = mat

    return out


###############################################################################
# Constant overlap-add processing class


def _check_store(store):
    if isinstance(store, np.ndarray):
        store = [store]
    if isinstance(store, (list, tuple)) and all(isinstance(s, np.ndarray)
                                                for s in store):
        store = _Storer(*store)
    if not callable(store):
        raise TypeError('store must be callable, got type %s'
                        % (type(store),))
    return store


class COLA:
    """Constant overlap-add processing helper.

    Parameters
    ----------
    process : callable
        A function that takes a chunk of input data with shape
        `(n_channels, n_samples)` and processes it.
    store : callable | ndarray
        A function that takes a completed chunk of output data.
        Can also be an `ndarray`, in which case it is treated as the
        output data in which to store the results.
    n_total : int
        The total number of samples.
    n_samples : int
        The number of samples per window.
    n_overlap : int
        The overlap between windows.
    window : str
        The window to use. Default is "hann".
    tol : float
        The tolerance for COLA checking.
    n_jobs : int
        The number of jobs to run in parallel.
    verbose : bool
        If True, print a message when the COLA condition is not met.

    Notes
    -----
    This will process data using overlapping windows to achieve a constant
    output value. For example, for ``n_total=27``, ``n_samples=10``,
    ``n_overlap=5`` and ``window='triang'``
    ::

        1 _____               _______
          |    \\   /\\   /\\   /
          |     \\ /  \\ /  \\ /
          |      x    x    x
          |     / \\  / \\  / \\
          |    /   \\/   \\/   \\
        0 +----|----|----|----|----|-
          0    5   10   15   20   25

    This produces four windows: the first three are the requested length
    (10 samples) and the last one is longer (12 samples). The first and last
    window are asymmetric.
    """

    def __init__(self, process, store, n_total, n_samples, n_overlap, sfreq,
                 window='hann', tol=1e-10, *, verbose=None):
        n_samples = ensure_int(n_samples, 'n_samples')
        n_overlap = ensure_int(n_overlap, 'n_overlap')
        n_total = ensure_int(n_total, 'n_total')
        if n_samples <= 0:
            raise ValueError('n_samples must be > 0, got %s' % (n_samples,))
        if n_overlap < 0:
            raise ValueError('n_overlap must be >= 0, got %s' % (n_overlap,))
        if n_total < 0:
            raise ValueError('n_total must be >= 0, got %s' % (n_total,))
        self._n_samples = int(n_samples)
        self._n_overlap = int(n_overlap)
        del n_samples, n_overlap
        if n_total < self._n_samples:
            raise ValueError('Number of samples per window (%d) must be at '
                             'most the total number of samples (%s)'
                             % (self._n_samples, n_total))
        if not callable(process):
            raise TypeError('process must be callable, got type %s'
                            % (type(process),))
        self._process = process
        self._step = self._n_samples - self._n_overlap
        self._store = _check_store(store)
        self._idx = 0
        self._in_buffers = self._out_buffers = None

        # Create our window boundaries
        window_name = window if isinstance(window, str) else 'custom'
        self._window = get_window(window, self._n_samples,
                                  fftbins=(self._n_samples - 1) % 2)
        self._window /= _check_cola(self._window, self._n_samples, self._step,
                                    window_name, tol=tol)
        self.starts = np.arange(0, n_total - self._n_samples + 1, self._step)
        self.stops = self.starts + self._n_samples
        delta = n_total - self.stops[-1]
        self.stops[-1] = n_total
        sfreq = float(sfreq)
        pl = 's' if len(self.starts) != 1 else ''
        if verbose:
            logger.info('    Processing %4d data chunk%s of (at least) %0.1f '
                        'sec with %0.1f sec overlap and %s windowing'
                        % (len(self.starts), pl, self._n_samples / sfreq,
                           self._n_overlap / sfreq, window_name))
        del window, window_name
        if delta > 0 and verbose:
            logger.info('    The final %0.3f sec will be lumped into the '
                        'final window' % (delta / sfreq,))

    @property
    def _in_offset(self):
        """Compute from current processing window start and buffer len."""
        return self.starts[self._idx] + self._in_buffers[0].shape[-1]

    def feed(self, *datas, verbose=None, **kwargs):
        """Pass in a chunk of data."""
        # Append to our input buffer
        if self._in_buffers is None:
            self._in_buffers = [None] * len(datas)
        if len(datas) != len(self._in_buffers):
            raise ValueError('Got %d array(s), needed %d'
                             % (len(datas), len(self._in_buffers)))
        for di, data in enumerate(datas):
            if not isinstance(data, np.ndarray) or data.ndim < 1:
                raise TypeError('data entry %d must be an 2D ndarray, got %s'
                                % (di, type(data),))
            if self._in_buffers[di] is None:
                # In practice, users can give large chunks, so we use
                # dynamic allocation of the in buffer. We could save some
                # memory allocation by only ever processing max_len at once,
                # but this would increase code complexity.
                self._in_buffers[di] = np.empty(
                    data.shape[:-1] + (0,), data.dtype)
            if data.shape[:-1] != self._in_buffers[di].shape[:-1] or \
                    self._in_buffers[di].dtype != data.dtype:
                raise TypeError('data must dtype %s and shape[:-1]==%s, '
                                'got dtype %s shape[:-1]=%s'
                                % (self._in_buffers[di].dtype,
                                   self._in_buffers[di].shape[:-1],
                                   data.dtype, data.shape[:-1]))
            # logger.debug('    + Appending %d->%d'
            #              % (self._in_offset, self._in_offset + data.shape[
            #              -1]))
            self._in_buffers[di] = np.concatenate(
                [self._in_buffers[di], data], -1)
            if self._in_offset > self.stops[-1]:
                raise ValueError('data (shape %s) exceeded expected total '
                                 'buffer size (%s > %s)'
                                 % (data.shape, self._in_offset,
                                    self.stops[-1]))
        # preallocate data to chunks
        data_chunks = map(lambda x, y: data[x:y], self.starts, self.stops)
        out_chunks = map(lambda d: self._process(d, **kwargs), data_chunks)

        # overlap add to buffer
        while self._idx < len(self.starts) and \
                self._in_offset >= self.stops[self._idx]:
            start, stop = self.starts[self._idx], self.stops[self._idx]
            this_len = stop - start
            this_window = self._window.copy()
            if self._idx == len(self.starts) - 1:
                this_window = np.pad(
                    self._window, (0, this_len - len(this_window)), 'constant')
                for offset in range(self._step, len(this_window), self._step):
                    n_use = len(this_window) - offset
                    this_window[offset:] += self._window[:n_use]
            if self._idx == 0:
                for offset in range(self._n_samples - self._step, 0,
                                    -self._step):
                    this_window[:offset] += self._window[-offset:]
            # logger.debug('    * Processing %d->%d' % (start, stop))
            this_proc = [in_[..., :this_len].copy()
                         for in_ in self._in_buffers]
            if not all(proc.shape[-1] == this_len == this_window.size
                       for proc in this_proc):
                raise RuntimeError('internal indexing error')
            outs = next(out_chunks)
            if self._out_buffers is None:
                max_len = np.max(self.stops - self.starts)
                self._out_buffers = [np.zeros(o.shape[:-1] + (max_len,),
                                              o.dtype) for o in outs]
            for oi, out in enumerate(outs):
                out *= this_window
                self._out_buffers[oi][..., :stop - start] += out
            self._idx += 1
            if self._idx < len(self.starts):
                next_start = self.starts[self._idx]
            else:
                next_start = self.stops[-1]
            delta = next_start - self.starts[self._idx - 1]
            for di in range(len(self._in_buffers)):
                self._in_buffers[di] = self._in_buffers[di][..., delta:]
            # logger.debug('    - Shifting input/output buffers by %d samples'
            #              % (delta,))
            self._store(*[o[..., :delta] for o in self._out_buffers])
            for ob in self._out_buffers:
                ob[..., :-delta] = ob[..., delta:]
                ob[..., -delta:] = 0.


def _check_cola(win, nperseg, step, window_name, tol=1e-10):
    """Check whether the Constant OverLap Add (COLA) constraint is met."""
    # adapted from SciPy
    binsums = np.sum([win[ii * step:(ii + 1) * step]
                      for ii in range(nperseg // step)], axis=0)
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]
    const = np.median(binsums)
    deviation = np.max(np.abs(binsums - const))
    if deviation > tol:
        raise ValueError('segment length %d with step %d for %s window '
                         'type does not provide a constant output '
                         '(%g%% deviation)'
                         % (nperseg, step, window_name,
                            100 * deviation / const))
    return const


class _Storer(object):
    """Store data in chunks."""

    def __init__(self, *outs, picks=None):
        for oi, out in enumerate(outs):
            if not isinstance(out, np.ndarray) or out.ndim < 1:
                raise TypeError('outs[oi] must be >= 1D ndarray, got %s'
                                % (out,))
        self.outs = outs
        self.idx = 0
        self.picks = picks

    def __call__(self, *outs):
        if (len(outs) != len(self.outs) or
                not all(out.shape[-1] == outs[0].shape[-1] for out in outs)):
            raise ValueError('Bad outs')
        idx = (Ellipsis,)
        if self.picks is not None:
            idx += (self.picks,)
        stop = self.idx + outs[0].shape[-1]
        idx += (slice(self.idx, stop),)
        for o1, o2 in zip(self.outs, outs):
            o1[idx] = o2
        self.idx = stop
