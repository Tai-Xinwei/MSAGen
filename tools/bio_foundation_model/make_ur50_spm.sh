#!/bin/bash
set -euo pipefail
set -x

raw_data='/blob/lihe/data/protein/uniref50_2018_03.train.seqs.shorten'

# run sentencepiece
nCodes=16384

spm_train --input=$raw_data \
    --input_sentence_size=1000000 --shuffle_input_sentence=true \
    --model_prefix=/tmp/ur50bpe \
    --vocab_size=$nCodes \
    --character_coverage=1.0 \
    --pad_id=3 \
    --add_dummy_prefix=false \
    --train_extremely_large_corpus=true

spm_export_vocab --model=/tmp/ur50bpe.model | cut -f1 > /tmp/ur50bpe.vocab


# copy to dest
mkdir -p /blob/shufxi/data/biofm/ur50bpe/
cp /tmp/ur50bpe.model /blob/shufxi/data/biofm/ur50bpe/
cp /tmp/ur50bpe.vocab /blob/shufxi/data/biofm/ur50bpe/
