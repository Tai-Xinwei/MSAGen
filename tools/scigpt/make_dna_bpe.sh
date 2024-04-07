#!/bin/bash

raw_data='/blob/yinxia/wu2/shared/SFM/SFM.overall.data/DNA/DNAseq.9Btoken.txt'
bpe_input='/tmp/dna.txt'
cat $raw_data | shuf -n 1000000 | sed -e 's/<DNA>//g' -e 's/<\/DNA>//g' > $bpe_input

# run sentencepiece
bpe_input='/tmp/dna.txt'
nCodes=1024

spm_train --input=$bpe_input \
    --model_prefix=/tmp/bpe \
    --vocab_size=$nCodes \
    --character_coverage=1.0 \
    --add_dummy_prefix=false \
    --bos_id=-1 --eos_id=-1

spm_export_vocab --model=/tmp/bpe.model | cut -f1 > /tmp/bpe.vocab

# copy to dest
mkdir -p /blob/shufxi/data/scigpt/dnabpe/
cp /tmp/bpe.model /blob/shufxi/data/scigpt/dnabpe/
cp /tmp/bpe.vocab /blob/shufxi/data/scigpt/dnabpe/
