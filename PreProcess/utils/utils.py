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
PathLike = TypeVar("PathLike", str, PL)


# plotting funcs

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


def parallelize(func: callable, par_var: Iterable, n_jobs: int = None, *args,
                **kwargs) -> list:
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
    from psutil import virtual_memory
    ram_per = virtual_memory()[3]/cpu_count()
    return ram_per
