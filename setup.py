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
        language="c++",  # can be "c" or "c++"
    ),
]

setup(
    name="Mixup Module",
    ext_modules=cythonize(extensions),
)