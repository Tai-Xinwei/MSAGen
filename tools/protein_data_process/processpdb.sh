#!/bin/bash

nohup python structure2lmdb.py processpdb --mmcif-dir /mnt/sfm/psm/20240101_PDB_Training_Data/20240101_snapshot/mmCIF --chem-comp-file /mnt/sfm/psm/20240101_PDB_Training_Data/20240101_snapshot/components.cif.gz --output-lmdb /mnt/sfm/psm/20240101_PDB_Training_Data/20240101_snapshot_mmCIF.lmdb --num-workers 32 1>stdout 2>stderr &
