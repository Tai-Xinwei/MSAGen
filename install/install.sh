#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "pip install start"
# # # python=3.9, cuda 11.7
pip install -r install/requirements.txt

pip install torch-scatter==2.1.2 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.18 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip install torch-geometric==2.3.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster==1.6.2 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
python setup_cython.py build_ext --inplace
