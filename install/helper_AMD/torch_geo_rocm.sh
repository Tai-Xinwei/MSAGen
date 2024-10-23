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

function install_torchgeo_torch23() {
  if python -c "import torch_geometric" &> /dev/null; then
    echo "Skipping torch geometric installation as it is already installed."
  else
    cd /tmp
    pv=$(python --version | cut -d '.' -f 2)
    wget -q https://github.com/Looong01/pyg-rocm-build/releases/download/5/torch-2.2-rocm-5.7-py3${pv}-linux_x86_64.zip
    unzip torch-2.2-rocm-5.7-py3${pv}-linux_x86_64.zip
    cd torch-2.2-rocm-5.7-py3${pv}-linux_x86_64
    pip install --disable-pip-version-check \
        --no-cache-dir --no-build-isolation ./*
    cd $ROOT
    rm -rf /tmp/torch-2.2-rocm-5.7-py3${pv}-linux_x86_64*
  fi
}

function install_torchgeo_torch24() {
  if python -c "import torch_geometric" &> /dev/null; then
    echo "Skipping torch geometric installation as it is already installed."
  else
    cd /tmp
    pv=$(python --version | cut -d '.' -f 2)
    torch=2.4-rocm-6.1
    pv=$(python --version | cut -d '.' -f 2)
    wget -q https://github.com/Looong01/pyg-rocm-build/releases/download/6/torch-${torch}-py3${pv}-linux_x86_64.zip
    unzip torch-${torch}-py3${pv}-linux_x86_64.zip -d torch-${torch}-py3${pv}-linux_x86_64
    cd torch-${torch}-py3${pv}-linux_x86_64
    pip install --disable-pip-version-check \
        --no-cache-dir --no-build-isolation ./*
    cd $ROOT
    rm -rf /tmp/torch-${torch}-py3${pv}-linux_x86_64*
  fi
}

function install_torchgeo_torch25() {
  if python -c "import torch_geometric" &> /dev/null; then
    echo "Skipping torch geometric installation as it is already installed."
  else
    cd /tmp
    pv=$(python --version | cut -d '.' -f 2)
    torch=2.5-rocm-6.2
    wget -q https://github.com/Looong01/pyg-rocm-build/releases/download/6/torch-${torch}-py3${pv}-linux_x86_64.zip
    unzip torch-${torch}-py3${pv}-linux_x86_64.zip -d torch-${torch}-py3${pv}-linux_x86_64
    cd torch-${torch}-py3${pv}-linux_x86_64
    pip install --disable-pip-version-check \
        --no-cache-dir --no-build-isolation ./*
    cd $ROOT
    rm -rf /tmp/torch-${torch}-py3${pv}-linux_x86_64*
  fi
}

function install_torchgeo_torch26() {
  if python -c "import torch_geometric" &> /dev/null; then
    echo "Skipping torch geometric installation as it is already installed."
  else
    cd /tmp
    pv=$(python --version | cut -d '.' -f 2)
    # MSR-AI4S repo -- login to GH via MS account needed!
    wget -q https://github.com/msr-ai4science/pyg_amd/releases/download/1/torch-2.6-rocm-6.2-py3${pv}-linux_x86_64.zip
    unzip torch-2.6-rocm-6.2-py3${pv}-linux_x86_64.zip
    cd torch-2.6-py3${pv}-linux_x86_64
    pip install --disable-pip-version-check \
        --no-cache-dir --no-build-isolation ./*whl
    cd $ROOT
    rm -rf /tmp/torch-2.6-py3${pv}-linux_x86_64*
  fi
}

function main() {
  echo "Passed argument: $1"

  case $1 in
    "torch-2.3")
      install_torchgeo_torch23
      ;;
    "torch-2.4")
      install_torchgeo_torch24
      ;;
    "torch-2.5")
      install_torchgeo_torch25
      ;;
    "torch-2.6")
      install_torchgeo_torch26
      ;;
    *)
      echo "Invalid PyTorch version. Supported versions: 2.3, 2.4, 2.5, 2.6"
      exit 1
      ;;
  esac

  check_environment $1
}

main "$@"
