import operator
import os.path as op
from os import PathLike as PL
from os import environ
from typing import TypeVar, Iterable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from mne.utils import config

HOME = op.expanduser("~")
LAB_root = op.join(HOME, "Box", "CoganLab")
PathLike = Union[str, PL]


def ensure_int(x, name='unknown', must_be='an int', *, extra=''):
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
    """
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


def parallelize(func: callable, par_var: Iterable, n_jobs: int = None, *args,
                **kwargs) -> list:
    """Parallelize a function to run on multiple processors.

    Parameters
    ----------
    func : callable
        The function to parallelize
    par_var : Iterable
        The iterable to parallelize over
    n_jobs : int
        The number of jobs to run in parallel. If None, will use all
        available cores. If -1, will use all available cores.
    *args
        Additional arguments to pass to the function
    **kwargs
        Additional keyword arguments to pass to the function

    Returns
    -------
    list
        The output of the function for each element in par_var
    """
    if 'n_jobs' in kwargs.keys():
        n_jobs = kwargs.pop('n_jobs')
    elif n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == -1:
        n_jobs = cpu_count()
    settings = dict(verbose=0, prefer='threads')
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
        x_, *args, **kwargs)for x_ in par_var)
    return data_new


def get_mem() -> Union[float, int]:
    """Get the amount of memory to use for parallelization."""
    from psutil import virtual_memory
    ram_per = virtual_memory()[3]/cpu_count()
    return ram_per
