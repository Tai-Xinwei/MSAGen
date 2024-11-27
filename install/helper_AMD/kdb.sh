#!/usr/bin/env bash
# https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.0/install/3rd-party/pytorch-install.html

set -e

function install_kdb() {
  wget -q https://raw.githubusercontent.com/wiki/ROCm/pytorch/files/install_kdb_files_for_pytorch_wheels.sh -O /tmp/install_kdb_files_for_pytorch_wheels.sh
  cd /tmp
  sed 's#opt\/#/opt/#g' install_kdb_files_for_pytorch_wheels.sh > install_kdb_files_for_pytorch_wheels_edit.sh
  export GFX_ARCH="$1"
  export ROCM_VERSION=6.2
  bash install_kdb_files_for_pytorch_wheels.sh
  rm -f install_kdb_files_for_pytorch_wheels.sh
}

function main() {
  echo "Passed argument: $1"

  install_kdb $1
}

main "$@"
