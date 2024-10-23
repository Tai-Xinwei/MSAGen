#!/usr/bin/env bash
# https://github.com/ROCmSoftwarePlatform/flash-attention.git

set -e

function check_environment() {
  echo "Conda env: $CONDA_PREFIX"
  echo "Python version: $(python --version)"
  echo "PyTorch version: $(python -c 'import torch;print(torch.__version__)')"
  python -c "import flash_attn" &> /dev/null && \
    echo "flash attention: Found" || \
    echo "flash attention: Not Found"
}

function install_flashattn() {
  if python -c "import flash_attn" &> /dev/null; then
    echo "Skipping flash attention installation as it is already installed."
  else
    git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/flash-attention
    cd /tmp/flash-attention
    export GPU_ARCHS="$1"
    export PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
    python setup.py install
    cd $ROOT
    rm -rf /tmp/flash-attention
  fi
}

function main() {
  echo "Passed argument: $1"

  install_flashattn "$1"
  check_environment "$1"
}

main "$@"
