#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
echo 'print path'
pwd
ls /nfs/

echo "start data copy"

mkdir /mnt/amlt_code/ur50_23_bpe_pack1536.lmdb
cp -r /nfs/psmdata/ur50_23_bpe_pack1536.lmdb/* /mnt/amlt_code/ur50_23_bpe_pack1536.lmdb/

mkdir /mnt/amlt_code/uniref50_pack1024_valid.lmdb
# cp -r /sfm/psm/afdb/uniref50_pack1024_valid.lmdb/ /nfs/psmdata/uniref50_pack1024_valid.lmdb/
cp -r /nfs/psmdata/uniref50_pack1024_valid.lmdb/* /mnt/amlt_code/uniref50_pack1024_valid.lmdb/

echo "data copied"
