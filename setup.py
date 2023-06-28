from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("ieeg.calc.arraydict", ["ieeg/calc/arraydict.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    name="ieeg",
    packages=find_packages(),
    ext_modules=cythonize(extensions)
)