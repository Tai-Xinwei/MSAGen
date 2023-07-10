# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.command import egg_info


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements("requirements/requirements.txt")
extras_require = {"train": fetch_requirements("requirements/requirements_train.txt")}

setup(
    name="SFM",
    version="0.0.1",
    description="CodeBase for Scientific Foundation Model",
    author="SFM Team",
    author_email="sfm.core@microsoft.com",
    install_requires=install_requires,
    extras_require=extras_require,
    ext_modules=cythonize("sfm/data/mol_data/algos.pyx"),
    include_dirs=[numpy.get_include()],
)


if __name__ == "__main__":
    pass
