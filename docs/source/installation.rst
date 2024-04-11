.. A4SFramework documentation master file, created by
   sphinx-quickstart on Mon Sep 25 05:01:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _installation:

Installation Guide
==================

This is a guide to install A4SFramework. Currently A4SFramework supports intallation on Linux only.

Linux / WSL
~~~~~

On Linux, Graphormer can be easily installed with the install.sh script with prepared python environments.

1. Please use Python3.10 or later for A4SFramework. It is recommended to create a virtual environment with `conda <https://docs.conda.io/en/latest/>`__ or `virtualenv <https://virtualenv.pypa.io/en/latest/>`__.
For example, to create and activate a conda environment with Python3.10

.. code::

    conda env create -f ./install/py310-torch2.2.2-cuda12.1.yaml -n A4SFramework
    conda activate A4SFramework

2. Run the following commands:

.. code::

    git clone --recursive git@github.com:msr-ai4science/feynman.git
    cd feynman/projects/SFM/experimental/SFM_framework
    python setup_cython.py build_ext --inplace
