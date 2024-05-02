#!/usr/bin/env bash
set -e

HERE="$(dirname "$(readlink -f "$0")")"

function check_environment() {
  echo "CUDA version: $(python -c 'import torch;print(torch.version.cuda)')"
  echo "NVIDIA NCCL version: $(python -c 'import torch;print(torch.cuda.nccl.version())')"
  echo "Conda env: $CONDA_PREFIX"
  echo "Python version: $(python --version)"
  echo "PyTorch version: $(python -c 'import torch;print(torch.__version__)')"
  python -c "from apex.optimizers import FusedAdam" &> /dev/null && \
    echo "NVIDIA Apex: Found" || \
    echo "NVIDIA Apex: Not Found"
}

function install_apex() {
  if python -c "from apex.optimizers import FusedAdam" &> /dev/null; then
    echo "Skipping NVIDIA Apex installation as it is already installed."
  else
    git clone https://github.com/NVIDIA/apex /tmp/apex
    cd /tmp/apex
    git checkout 23.08
    pip install -v --disable-pip-version-check \
        --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./
    cd $ROOT
    rm -rf /tmp/apex
  fi
}

function main() {
  install_apex
  check_environment
}

main "$@"
