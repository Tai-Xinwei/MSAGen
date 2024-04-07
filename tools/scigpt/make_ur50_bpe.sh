#!/bin/bash

raw_data='/blob/yinxia/wu2/shared/SFM/SFM.overall.data/ur50/uniref50_2018_03.train.seqs.pended.new.txt'
bpe_input='/tmp/ur50.train.seqs.pended.new.txt'

# clean <protein> and </protein>, random sample 1M and save to a temp file
cat $raw_data | shuf -n 1000000 | sed -e 's/<protein>//g' -e 's/<\/protein>//g' > $bpe_input

# run sentencepiece
nCodes=4096

spm_train --input=$bpe_input \
    --model_prefix=/tmp/bpe \
    --vocab_size=$nCodes \
    --character_coverage=1.0 \
    --add_dummy_prefix=false \
    --bos_id=-1 --eos_id=-1

spm_export_vocab --model=/tmp/bpe.model | cut -f1 > /tmp/bpe.vocab


# copy to dest
mkdir -p /blob/shufxi/data/scigpt/ur50bpe/
cp /tmp/bpe.model /blob/shufxi/data/scigpt/ur50bpe/
cp /tmp/bpe.vocab /blob/shufxi/data/scigpt/ur50bpe/
