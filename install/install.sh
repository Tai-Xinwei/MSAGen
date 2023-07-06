#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "pip install start"
# # # python=3.9, cuda 11.7

pip install deepspeed==0.9.5
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install tensorboard
pip install numba

pip install torch-scatter==2.1.0 -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
pip install torch-sparse==0.6.16 -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
pip install torch-geometric==2.2.0
pip install torch_cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html

pip install mpi4py
pip install Cython==0.29.32
pip install torchvision 

# pip install pytorch_forecasting
pip install networkx --user
pip install lmdb --user
python setup_cython.py build_ext --inplace

pip install mlflow azureml-mlflow --user
pip install transformers==4.30.1
pip install peft==0.3.0
pip install sentencepiece==0.1.99
pip install torch==2.0.1 --upgrade

