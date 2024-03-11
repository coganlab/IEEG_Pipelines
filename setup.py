from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
from setuptools import setup, find_packages
import os.path as op


# Read requirements.txt
with open(op.join('envs', 'requirements.txt')) as f:
    requirements = f.read().splitlines()

extensions = [
    Extension(
        "ieeg.calc.mixup",  # the module name exposed to python
        ["ieeg/calc/mixup.pyx"],  # the Cython source file
        include_dirs=[np.get_include(), 'ieeg/calc'],  # includes for numpy
        extra_compile_args=["-O3"],  # compile optimization flag
        language="c",  # can be "c" or "c++"
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "ieeg.calc.concat",  # the module name exposed to python
        ["ieeg/calc/concat.pyx"],
        # the Cython source file
        include_dirs=[np.get_include(), 'ieeg/calc'],  # includes for numpy
        extra_compile_args=["-O3"],  # compile optimization flag
        language="c",  # can be "c" or "c++"
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "ieeg.calc.permgt",  # the module name exposed to python
        ["ieeg/calc/permgt.pyx"],
        # the Cython source file
        include_dirs=[np.get_include(), 'ieeg/calc'],  # includes for numpy
        extra_compile_args=["-O3"],  # compile optimization flag
        language="c",  # can be "c" or "c++"
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "ieeg.timefreq.hilbert",  # the module name exposed to python
        ["ieeg/timefreq/hilbert.pyx"],
        # the Cython source file
        include_dirs=[np.get_include(), 'ieeg/timefreq'],  # includes for numpy
        extra_compile_args=["-O3"],  # compile optimization flag
        language="c",  # can be "c" or "c++"
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    name='ieeg',
    version='0.1',
    packages=find_packages(),
    description='A Python package for iEEG data processing.',
    author='Aaron Earle-Richardson',
    author_email='ae166@duke.edu',
    url='https://github.com/coganlab/IEEG_Pipelines',
    install_requires=requirements,
    ext_modules=cythonize(extensions, build_dir="build"),
    package_data={'': ['*.pyx', '*.pyd', '*.c', '*.h']},
    include_package_data=True,
)