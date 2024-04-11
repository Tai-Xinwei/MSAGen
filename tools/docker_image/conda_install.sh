#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

# TODO: test python 3.10 / 3.11
eval "$(conda shell.bash hook)" > /dev/null 2>&1 && \
    conda env remove -n sfm -y > /dev/null 2>&1 && \
    conda create -yn sfm python=3.10 > /dev/null 2>&1 && \
    conda activate sfm && \
    echo -e "${GREEN}SUCCESS${NC}: conda environment created and activated" || \
    echo -e "${RED}ERROR${NC}: Failed to create and activate conda environment"
# upgrade pip
pip install pip -U > /dev/null 2>&1 && \
    echo -e "${GREEN}SUCCESS${NC}: pip upgraded" || \
    (echo -e "${RED}ERROR${NC}: Failed to upgrade pip"; exit 1)

# Modify this line from pytorch.org to install other versions of PyTorch for legacy code
# conda install -n sfm -y pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia > /dev/null 2>&1 && \
#     echo -e "${GREEN}SUCCESS${NC}: pytorch installed" || \
#     echo -e "${RED}ERROR${NC}: Failed to install pytorch"

# the latest version is pytorch 2.2.2 currently
conda install pytorch=2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia > /dev/null 2>&1 && \
    echo -e "${GREEN}SUCCESS${NC}: pytorch installed" || \
    echo -e "${RED}ERROR${NC}: Failed to install pytorch"


conda install -n sfm -y conda-forge::rdkit \
    conda-forge::biopython \
    conda-forge::ogb \
    conda-forge::wandb \
    conda-forge::networkx \
    conda-forge::python-lmdb \
    conda-forge::dm-tree \
    conda-forge::tensorboard \
    conda-forge::loguru \
    conda-forge::transformers \
    conda-forge::mendeleev \
    conda-forge::sentencepiece \
    conda-forge::peft \
    conda-forge::setuptools-scm \
    defaults::cython \
    pyg::pyg > /dev/null 2>&1 && \
    echo -e "${GREEN}SUCCESS${NC}: conda packages installed" || \
    echo -e "${RED}ERROR${NC}: Failed to install conda packages"

# other packages that cannot use conda
# torch-tb-profiler: does not have a conda package in any channel.
# deepspeed: conda automatically uses CPU version since only CUDA 12.0 version is available.
# packaging ninja are for flash attention according to its github page.
# sacremoses is required by BioGptTokenizer.
pip install torch-tb-profiler deepspeed packaging ninja sacremoses > /dev/null 2>&1 && \
    echo -e "${GREEN}SUCCESS${NC}: torch-tb-profiler deepspeed packaging ninja sacremoses installed" || \
    echo -e "${RED}ERROR${NC}: Failed to install torch-tb-profiler deepspeed packaging ninja sacremoses"
# set nvcc threads when building, ref: https://github.com/Dao-AILab/flash-attention/blob/85881f547fd1053a7b4a2c3faad6690cca969279/setup.py#L86
# comment out flash-atten since PyTorch > 2.2 has built-in support. Ref: https://pytorch.org/blog/pytorch2-2/
# if some legacy code still needs flash-atten, uncomment the following lines and rebuild the image or add it to job commands.
# NVCC_THREADS=0 pip install flash-attn --no-build-isolation > /dev/null 2>&1 && \
#     echo -e "${GREEN}SUCCESS${NC}: flash-attn installed" || \
#     echo -e "${RED}ERROR${NC}: Failed to install flash-attn"

# install apex. Ref: https://github.com/NVIDIA/apex#quick-start
# This step is super slow...
cwd=$(pwd)
git clone https://github.com/NVIDIA/apex /tmp/apex > /dev/null 2>&1 && cd /tmp/apex || \
    echo -e "${RED}ERROR${NC}: Failed to git clone NVIDIA Apex"
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
# MAX_JOBS is the key for speed up compiling...
MAX_JOBS=0 pip install -v --disable-pip-version-check \
    --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./ > /dev/null 2>&1 && \
    echo -e "${GREEN}SUCCESS${NC}: apex installed" || \
    echo -e "${RED}ERROR${NC}: Failed to pip install NVIDIA Apex"
cd $cwd
rm -rf /tmp/apex

# version check scripts
echo "Version check:"
echo -e "Conda version: ${GREEN}$(conda --version)${NC}"
echo -e "Python version: ${GREEN}$(python --version)${NC}"
echo -e "pip version: ${GREEN}$(pip --version)${NC}"
echo -e "PyTorch version: ${GREEN}$(python -c 'import torch;print(torch.__version__)')${NC}"
echo -e "CUDA version: ${GREEN}$(python -c 'import torch;print(torch.version.cuda)')${NC}"
echo -e "NVIDIA NCCL version: ${GREEN}$(python -c 'import torch;print(torch.cuda.nccl.version())')${NC}"
python -c 'from apex.optimizers import FusedAdam' > /dev/null 2>&1 && \
    echo -e "NVIDIA Apex: ${GREEN}installed successfully${NC}" || \
    echo -e "NVIDIA Apex: ${RED}installation failed${NC}"
