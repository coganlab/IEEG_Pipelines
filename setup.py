from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "mixup",  # the module name exposed to python
        ["ieeg/calc/mixup.pyx"],  # the Cython source file
        include_dirs=[np.get_include(), 'ieeg/calc'],  # includes for numpy
        extra_compile_args=["-O3"],  # compile optimization flag
        language="c",  # can be "c" or "c++"
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        "concat",  # the module name exposed to python
        ["ieeg/calc/concat.pyx"],
        # the Cython source file
        include_dirs=[np.get_include(), 'ieeg/calc'],  # includes for numpy
        extra_compile_args=["-O3"],  # compile optimization flag
        language="c",  # can be "c" or "c++"
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    name="C Optimized Functions",
    ext_modules=cythonize(extensions),
)