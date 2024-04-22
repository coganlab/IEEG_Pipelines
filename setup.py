from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
from setuptools import setup, find_packages
import os.path as op
from os import getcwd, sep
import sys


_numpy_abs = np.get_include()  # get the numpy include path


def fix_path(path):
    dirs = list(reversed(path.split(sep)))
    i = 0
    j = 0
    while i + j < len(dirs):
        if dirs[i + j] == "..":
            j += 1
        elif j > 0:
            dirs.pop(i + j)
            dirs.pop(i)
            j -= 1
        else:
            i += 1
    return sep.join(reversed(dirs))

npymath_path = fix_path(op.join(_numpy_abs, '..', 'lib'))
npyrandom_path = fix_path(op.join(_numpy_abs, '..', '..', 'random', 'lib'))
lib_path = [npymath_path, npyrandom_path]
if sys.platform == 'win32':
    compile_args = ["/O2"]
elif sys.platform == 'linux':
    compile_args = ["-O3"]
else:
    raise NotImplementedError(f"Platform {sys.platform} not supported.")



kwargs = dict(include_dirs=[op.relpath(_numpy_abs, getcwd())],
              # includes for numpy
              library_dirs=lib_path,  # libraries to link
              libraries=["npyrandom", "npymath"],  # math library
              extra_compile_args=compile_args,  # compile optimization flag
              language="c",  # can be "c" or "c++"
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
              )

extensions = [
    Extension(
        "ieeg.calc.fast.*",  # the module name exposed to python
        ["ieeg/calc/fast/*.pyx"],  # the Cython source file
        **kwargs
    ),
    Extension(
        "ieeg.calc.fast.stats",  # the module name exposed to python
        ["ieeg/calc/fast/stats.c"],  # the C source file
        **kwargs
    ),
    Extension(
        "ieeg.timefreq.hilbert",  # the module name exposed to python
        ["ieeg/timefreq/hilbert.pyx"],  # the Cython source file
        **kwargs
    ),
]

setup(
    name='ieeg',
    version='0.1',
    packages=find_packages() + ['ieeg.calc.fast'],
    description='A Python package for iEEG data processing.',
    author='Aaron Earle-Richardson',
    author_email='ae166@duke.edu',
    url='https://github.com/coganlab/IEEG_Pipelines',
    ext_modules=cythonize(extensions, build_dir="build"),
)