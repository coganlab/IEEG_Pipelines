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
    compile_args = ["-O3", "-march=x86-64-v3", "-flto", "-fomit-frame-pointer", "-DNDEBUG", "-fopenmp"]
    link_args = ["-fopenmp"]
elif sys.platform == 'darwin':
    compile_args = ["-O3", "-march=x86-64-v3", "-flto", "-DNDEBUG"]
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
        "ieeg.timefreq._hilbert_kernel",
        ['ieeg/timefreq/_hilbert_kernel.c'],
        **kwargs
    ),
    Extension(
        "ieeg.timefreq._superlets_kernels",
        ['ieeg/timefreq/_superlets_kernels.c'],
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
    # Ship the CUDA-C source for the superlets GPU kernel as a data file
    # so cupy.RawKernel can read it at runtime (nvrtc compiles on first use).
    package_data={"ieeg.timefreq": ["*.cu"]},
    include_package_data=True,
    description='A Python package for iEEG data processing.',
    author='Aaron Earle-Richardson',
    author_email='ae166@duke.edu',
    url='https://github.com/coganlab/IEEG_Pipelines',
    ext_modules=extensions,
)
