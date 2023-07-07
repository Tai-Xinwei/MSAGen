import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize("src/data/mol_data/algos.pyx"),
    include_dirs=[numpy.get_include()],
)
