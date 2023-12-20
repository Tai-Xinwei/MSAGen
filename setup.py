# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command import egg_info


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements("requirements/requirements.txt")

cython_extensions = ["sfm/data/mol_data/algos.pyx", "sfm/data/data_utils_fast.pyx"]
# cython_extensions = [
#     Extension("sfm.data.mol_data.algos", ["sfm/data/mol_data/algos.pyx"]),
#     Extension("sfm.data.data_utils_fast", ["sfm/data/data_utils_fast.pyx"]),
# ]

setup(
    name="a4sframework",
    version="0.1.0",
    description="A4S Framework",
    author="MSR A4S team",
    author_email="sfm.core@microsoft.com",
    install_requires=install_requires,
    ext_modules=cythonize(cython_extensions, language_level="3"),
    extras_require={
        "dev": ["flake8", "pytest", "black==22.3.0"],
        "docs": ["sphinx", "sphinx-argparse"],
    },
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


if __name__ == "__main__":
    pass
