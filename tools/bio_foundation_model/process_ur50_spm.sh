#!/bin/bash
set -x
set -euo pipefail

python tools/bio_foundation_model/process_ur50_spm.py \
    --input /blob/shufxi/data/biofm/ur50bpe/uniref50_2018_03.valid.seqs.shorten.100k \
    --output /blob/shufxi/data/biofm/ur50bpe/valid.npy

python tools/bio_foundation_model/process_ur50_spm.py \
    --input /blob/lihe/data/protein/uniref50_2018_03.train.seqs.shorten \
    --output /blob/shufxi/data/biofm/ur50bpe/train.npy
