from setuptools.extension import Extension
import numpy as np
from setuptools import setup, find_packages
import os.path as op
from os import getcwd
import sys
import glob


_numpy_abs = np.get_include()  # get the numpy include path

def get_file_list(path, ext):
    all_files = glob.glob(op.join(path, f"*{ext}"))
    for file in all_files:
        elem = op.split(file)
        name = op.splitext(".".join(elem))[0]
        yield name, [file]


npymath_path = op.normpath(op.join(_numpy_abs, '..', 'lib'))
npyrandom_path = op.normpath(op.join(_numpy_abs, '..', '..', 'random', 'lib'))
lib_path = [npymath_path, npyrandom_path]
if sys.platform == 'win32':
    compile_args = ["/O2"]
elif sys.platform == 'linux':
    compile_args = ["-O3"]
else:
    raise NotImplementedError(f"Platform {sys.platform} not supported.")

try:
    _numpy_abs = op.relpath(_numpy_abs, getcwd())
except ValueError:
    pass


kwargs = dict(include_dirs=[_numpy_abs],
              # includes for numpy
              library_dirs=lib_path,  # libraries to link
              libraries=["npyrandom", "npymath"],  # math library
              extra_compile_args=compile_args,  # compile optimization flag
              language="c",  # can be "c" or "c++"
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
              )

try:
    from Cython.Build import cythonize
    if not op.exists("ieeg/timefreq/hilbert.pyx"):
        raise ImportError("Cython file not found.")
    extensions = [
        Extension(
            "ieeg.calc._fast.*",  # the module name exposed to python
            ['ieeg/calc/_fast/*.pyx'],  # the Cython source file
            **kwargs
        ),
        Extension(
            "ieeg.timefreq.hilbert",  # the module name exposed to python
            ['ieeg/timefreq/hilbert.pyx'],  # the Cython source file
            **kwargs
        )]
    extensions = cythonize(extensions)
except ImportError:
    USE_CYTHON = False
    print("Cython not found. Using C files.")
    if not op.exists("ieeg/timefreq/hilbert.c"):
        ValueError("C file not found.")

extensions = [
    Extension(name, source, **kwargs) for (name, source) in
    get_file_list("ieeg/calc/_fast", ".c")]
extensions += [
    Extension(
        "ieeg.timefreq.hilbert",  # the module name exposed to python
        ['ieeg/timefreq/hilbert.c'],  # the Cython source file
        **kwargs
    ),
]

setup(
    name='ieeg',
    version='0.4',
    packages=find_packages(
        where='.',
        include=['ieeg*'],
    ),
    package_dir={"": "."},
    description='A Python package for iEEG data processing.',
    author='Aaron Earle-Richardson',
    author_email='ae166@duke.edu',
    url='https://github.com/coganlab/IEEG_Pipelines',
    ext_modules=extensions,
)