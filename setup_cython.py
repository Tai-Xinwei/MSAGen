# -*- coding: utf-8 -*-
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = ["sfm/data/mol_data/algos.pyx", "sfm/data/data_utils_fast.pyx"]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
