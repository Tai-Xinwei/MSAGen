#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'


azcopy_path=$(find /tmp -maxdepth 1 -type d -name 'azcopy_linux_amd64*')
$azcopy_path/azcopy copy "$train_data_sas" "/tmp/psmdata/" --recursive

echo

echo "data copied"
