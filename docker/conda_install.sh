#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

export CUDA_HOME="/usr/local/cuda"
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"

# TODO: add support for python 3.10 / 3.11
eval "$(conda shell.bash hook)" && \
    conda env create -f /tmp/sfm-cuda.yaml -n sfm && \
    conda activate sfm && \
    echo -e "${GREEN}SUCCESS${NC}: conda environment created and activated" || \
    echo -e "${RED}ERROR${NC}: Failed to create and activate conda environment"

# other packages that cannot directly install with conda or pip
# megablocks[gg] is required by mixtral.
pip install megablocks[gg] && \
    echo -e "${GREEN}SUCCESS${NC}: megablocks[gg] installed" || \
    echo -e "${RED}ERROR${NC}: Failed to install megablocks[gg]"
# set nvcc threads when building, ref: https://github.com/Dao-AILab/flash-attention/blob/85881f547fd1053a7b4a2c3faad6690cca969279/setup.py#L86
# flash-atten v2 are supported with PyTorch > 2.2 but have a different API Ref: https://pytorch.org/blog/pytorch2-2/
NVCC_THREADS=0 pip install flash-attn --no-build-isolation && \
    echo -e "${GREEN}SUCCESS${NC}: flash-attn installed" || \
    echo -e "${RED}ERROR${NC}: Failed to install flash-attn"
# install apex. Ref: https://github.com/NVIDIA/apex#quick-start
# This step is super slow...
cwd=$(pwd)
git clone https://github.com/NVIDIA/apex /tmp/apex > /dev/null 2>&1 && cd /tmp/apex || \
    echo -e "${RED}ERROR${NC}: Failed to git clone NVIDIA Apex"
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
# TORCH_CUDA_ARCH_LIST must be set for compatibility reasons.
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
