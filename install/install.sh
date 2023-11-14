#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "pip install start"
# # # python=3.9, cuda 11.7
pip install deepspeed==0.10.0
pip install ogb==1.3.6
pip install rdkit-pypi==2021.9.3
pip install rdkit==2023.3.3
pip install tensorboard
pip install numba
# pip install mpi4py

pip install torch-scatter==2.1.2 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html --upgrade
pip install torch-sparse==0.6.18 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html --upgrade
pip install torch-geometric==2.3.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html --upgrade
pip install torch_cluster==1.6.2 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html --upgrade

pip install Cython==0.29.32 --upgrade
pip install torchvision==0.16.0 --upgrade
pip install torcheval

# pip install pytorch_forecasting
pip install networkx
pip install lmdb
python setup_cython.py build_ext --inplace

pip install mlflow azureml-mlflow
pip install torch==2.1.0 --upgrade --index-url https://download.pytorch.org/whl/cu118
pip install wandb
pip install loguru

# For generalist
pip install transformers==4.34.0
pip install peft
pip install sentencepiece
# conda install -c conda-forge openbabel

# For tamgent
pip install sacremoses

# For 3D AR
pip install mendeleev
