#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
echo 'print path'
pwd
ls /nfs/

echo "start data copy"

mkdir /mnt/amlt_code/AFDB50-plddt70.lmdb
cp -r /nfs/psmdat/AFDB50-plddt70.lmdb/* /mnt/amlt_code/AFDB50-plddt70.lmdb/

echo "data copied"
