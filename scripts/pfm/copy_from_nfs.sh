#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
echo 'print path'
pwd
ls /nfs/

echo "start data copy"

mkdir /mnt/amlt_code/ur50_msa_ppi_bpe_pack1536_train.lmdb
cp -r /nfs/ur50_msa_ppi_bpe_pack1536_train.lmdb/* /mnt/amlt_code/ur50_msa_ppi_bpe_pack1536_train.lmdb/

mkdir /mnt/amlt_code/uniref50_pack1024_valid.lmdb
cp -r /nfs/uniref50_pack1024_valid_1.lmdb/* /mnt/amlt_code/uniref50_pack1024_valid.lmdb/

echo "data copied"
