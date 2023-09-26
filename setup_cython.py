# -*- coding: utf-8 -*-
import sys

import numpy

# import order matters!
# see https://stackoverflow.com/questions/21594925/error-each-element-of-ext-modules-option-must-be-an-extension-instance-or-2-t

# isort:skip_file
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = ["sfm/data/mol_data/algos.pyx", "sfm/data/data_utils_fast.pyx"]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    include_dirs=[numpy.get_include()],
)
