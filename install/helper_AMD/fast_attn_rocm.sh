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
    git clone --recursive https://github.com/ROCmSoftwarePlatform/flash-attention.git /tmp/flash-attn
    cd /tmp/flash-attn
    export GPU_ARCHS="gfx90a" # mi2xx
    # export GPU_ARCHS="gfx941;gfx942" # mi3xx
    export PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
    pip install --disable-pip-version-check \
        --no-cache-dir --no-build-isolation .
    cd $ROOT
    rm -rf /tmp/flash-attn
  fi
}

function main() {
  install_flashattn
  check_environment
}

main "$@"
