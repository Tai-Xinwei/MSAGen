#!/usr/bin/bash
set -exuo pipefail

TRAIN_FILES="/blob/v-zequnliu/mix_pretrain/c4.500k,
        /blob/v-zequnliu/mix_pretrain/train.pubmed_15M_title_abs.detok.prot.1m,
        /blob/shufxi/data/ur50split/uniref50_2023_05.shorten.train.taged2.seqs.selected,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/antibody/oas-5m-sampled-formatted.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/wrapped_data/train_wrapped_seq.txt.prot.1.5m,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/wrapped_data/PMC_v1_wrapped.txt.prot,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/protein-text.nonwrap-updated.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/geneannot_protein_desc_v3.txt"


TARGET_FOLDER="/home/shufxi/scigptdata/v3.prot.2"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"
if [ ! -f "${TARGET_FOLDER}/train.npy" ]; then
        python -u tools/scigpt/make_binary.py \
        --input "${TRAIN_FILES}" \
        --output ${TARGET_FOLDER}/train.npy \
        --tokenizer_path /hai1/ds_dataset/llama2/llama-2-7b \
        --num_workers "${WORKERS}" \
        --seq_len 4096
fi
echo "done"
