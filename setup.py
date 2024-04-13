# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import numpy
import setuptools_scm
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command import egg_info


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


try:
    __version__ = setuptools_scm.get_version()
except Exception:
    __version__ = "0.0.1"

install_requires = [
    # manually copied from conda yaml
    "torch @ https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp39-cp39-linux_x86_64.whl",
    "torchvision==0.17.2",
    "torchaudio==2.2.2",
    "biopython==1.83",
    "ogb==1.3.6",
    "wandb==0.16.5",
    "networkx==3.2.1",
    "lmdb==1.4.1",
    "dm-tree==0.1.8",
    "tensorboard==2.16.2",
    "loguru==0.7.2",
    "transformers==4.39.3",
    "mendeleev==0.15.0",
    "sentencepiece==0.2.0",
    "peft==0.10.0",
    "setuptools-scm==8.0.4",
    "cython==3.0.10",
    "torch_geometric==2.5.2",
    "pyg_lib @ https://data.pyg.org/whl/torch-2.2.2%2Bcu121/pyg_lib-0.4.0%2Bpt22cu121-cp39-cp39-linux_x86_64.whl",
    "torch_scatter @ https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_cluster-1.6.3%2Bpt22cu121-cp39-cp39-linux_x86_64.whl",
    "torch_sparse @ https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_sparse-0.6.18%2Bpt22cu121-cp39-cp39-linux_x86_64.whl",
    "torch_spline_conv @ https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt22cu121-cp39-cp39-linux_x86_64.whl",
    "torch-tb-profiler==0.4.3",
    "deepspeed==0.14.0",
    "packaging==23.2",
    "ninja==1.11.1.1",
    "pip==24.0",
    "sacremoses==0.1.1",
]
cython_extensions = ["sfm/data/mol_data/algos.pyx", "sfm/data/data_utils_fast.pyx"]
# cython_extensions = [
#     Extension("sfm.data.mol_data.algos", ["sfm/data/mol_data/algos.pyx"]),
#     Extension("sfm.data.data_utils_fast", ["sfm/data/data_utils_fast.pyx"]),
# ]

setup(
    name="a4sframework",
    version=__version__,
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
