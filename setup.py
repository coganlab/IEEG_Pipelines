from setuptools.extension import Extension
import numpy as np
from setuptools import setup, find_packages
import os.path as op
from os import getcwd
import sys
import glob


_numpy_abs = np.get_include()  # get the numpy include path

def get_file_list(path, ext, exclude=None):
    all_files = glob.glob(op.join(path, f"*{ext}"))
    for file in all_files:
        if exclude and op.basename(file) in exclude:
            continue
        # Convert file path to module name by replacing path separators with dots
        name = op.splitext(file)[0].replace('\\', '.').replace('/', '.')
        yield name, [file]


npymath_path = op.normpath(op.join(_numpy_abs, '..', 'lib'))
npyrandom_path = op.normpath(op.join(_numpy_abs, '..', '..', 'random', 'lib'))
lib_path = [npymath_path, npyrandom_path]
if sys.platform == 'win32':
    compile_args = ["/O2", "/GL", "/arch:AVX2", "/DNDEBUG", "/openmp"]
    link_args = []  # MSVC handles this implicitly
elif sys.platform == 'linux':
    compile_args = ["-O3", "-march=native", "-flto", "-fomit-frame-pointer", "-DNDEBUG", "-fopenmp"]
    link_args = ["-fopenmp"]
elif sys.platform == 'darwin':
    compile_args = ["-O3", "-march=native", "-flto", "-DNDEBUG"]
    link_args = []
    raise NotImplementedError(f"Platform {sys.platform} not supported.")

try:
    _numpy_abs = op.relpath(_numpy_abs, getcwd())
except ValueError:
    pass


shared_dir = op.join(getcwd(), "ieeg", "calc", "_fast", "shared")
kwargs = dict(include_dirs=[_numpy_abs, shared_dir],
              library_dirs=lib_path,
              libraries=["npyrandom", "npymath"],
              extra_compile_args=compile_args,
              extra_link_args=link_args,
              language="c",
              define_macros=[]
              )

def _with_macros(macros):
    kw = dict(kwargs)
    kw["define_macros"] = macros
    return kw

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

exclude_fast = {"ufuncs.c"}
extensions = [
    Extension(name, source, **kwargs) for (name, source) in
    get_file_list("ieeg/calc/_fast", ".c", exclude=exclude_fast)]
extensions += [
    Extension(
        "ieeg.calc._fast.ufuncs",
        ["ieeg/calc/_fast/ufuncs.c", "ieeg/calc/_fast/shared/meanvar_core.c"],
        **_with_macros([("PY_ARRAY_UNIQUE_SYMBOL", "IEEG_ARRAY_API_UFUNCS")])
    ),
    Extension(
        "ieeg.timefreq.hilbert",  # the module name exposed to python
        ['ieeg/timefreq/hilbert.c'],  # the Cython source file
        **kwargs
    ),
    Extension(
        "ieeg.arrays.labeledarray",
        ["ieeg/arrays/labeledarray.c", "ieeg/arrays/labels.c", "ieeg/calc/_fast/shared/meanvar_core.c"],
        **_with_macros([("PY_ARRAY_UNIQUE_SYMBOL", "IEEG_ARRAY_API_LABELEDARRAY")])
    )
]

setup(
    name='ieeg',
    version='0.7.0',
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
