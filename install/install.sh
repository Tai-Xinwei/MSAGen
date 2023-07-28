#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "pip install start"
# # # python=3.9, cuda 11.7

pip install deepspeed==0.10.0
pip install ogb==1.3.6
pip install rdkit-pypi==2021.9.3
pip install tensorboard
pip install numba

pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html --upgrade
pip install torch-sparse==0.6.17 -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html --upgrade
pip install torch-geometric==2.3.0 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html --upgrade
pip install torch_cluster==1.6.1 -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html --upgrade

pip install Cython==0.29.32 --upgrade
pip install torchvision==0.15.2 --upgrade

# pip install pytorch_forecasting
pip install networkx
pip install lmdb
python setup_cython.py build_ext --inplace

pip install mlflow azureml-mlflow
pip install torch==2.0.1 --upgrade
pip install wandb

# For generalist
pip install transformers
pip install peft
pip install sentencepiece
