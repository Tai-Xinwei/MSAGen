#!/usr/bin/env bash
# https://github.com/Looong01/pyg-rocm-build

set -e

function check_environment() {
  echo "Conda env: $CONDA_PREFIX"
  echo "Python version: $(python --version)"
  echo "PyTorch version: $(python -c 'import torch;print(torch.__version__)')"
  python -c "import torch_geometric" &> /dev/null && \
    echo "torch geometric: Found" || \
    echo "torch geometric: Not Found"
}

function install_torchgeo() {
  if python -c "import torch_geometric" &> /dev/null; then
    echo "Skipping torch geometric installation as it is already installed."
  else
    cd /tmp
    wget https://github.com/Looong01/pyg-rocm-build/releases/download/5/torch-2.2-rocm-5.7-py311-linux_x86_64.zip
    unzip torch-2.2-rocm-5.7-py311-linux_x86_64.zip
    cd torch-2.2-rocm-5.7-py311-linux_x86_64
    pip install --disable-pip-version-check \
        --no-cache-dir --no-build-isolation ./*
    cd $ROOT
    rm -rf /tmp/torch-2.2-rocm-5.7-py311-linux_x86_64.zip
  fi
}

function main() {
  install_torchgeo
  check_environment
}

main "$@"
