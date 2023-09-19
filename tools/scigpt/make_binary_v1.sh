#!/usr/bin/bash
set -exuo pipefail

DATA_HOME="/blob/yinxia/wu2/shared/SFM/SFM.overall.data"

VAL_FILES="${DATA_HOME}/molecule/valid.pubchem.30M.can.tok.marked.pended.new.txt,
        ${DATA_HOME}/antibody/full_seq_rmdup.sample30m.valid.pended.new.txt,
        ${DATA_HOME}/ur50/uniref50_2018_03.valid.seqs.pended.new.txt,
        ${DATA_HOME}/material/all_materials.pended.valid.new.txt,
        ${DATA_HOME}/wrapped_data/valid_wrapped_seq.txt,
        /blob/v-zequnliu/mix_pretrain/valid.pubmed_15M_title_abs.detok"


TRAIN_FILES="${DATA_HOME}/molecule/train.pubchem.30M.can.tok.marked.pended.new.txt,
        ${DATA_HOME}/antibody/full_seq_rmdup.sample30m.train.pended.new.txt,
        ${DATA_HOME}/ur50/uniref50_2018_03.train.seqs.pended.new.txt,
        ${DATA_HOME}/material/all_materials.pended.train.new.txt,
        ${DATA_HOME}/wrapped_data/train_wrapped_seq.txt,
        /blob/v-zequnliu/mix_pretrain/train.pubmed_15M_title_abs.detok,
        /blob/v-zequnliu/mix_pretrain/c4"


TARGET_FOLDER="/blob/shufxi/data/scigpt/v1"
WORKERS=$(nproc)

echo "processing valid data"

python -u tools/scigpt/make_binary.py \
    --input "${VAL_FILES}" \
    --output ${TARGET_FOLDER}/valid.npy \
    --tokenizer_path /hai1/ds_dataset/llama2/llama-2-7b \
    --num_workers "${WORKERS}" \
    --seq_len 4096

echo "processing train data"
python -u tools/scigpt/make_binary.py \
    --input "${TRAIN_FILES}" \
    --output ${TARGET_FOLDER}/train.npy \
    --tokenizer_path /hai1/ds_dataset/llama2/llama-2-7b \
    --num_workers "${WORKERS}" \
    --seq_len 4096

echo "done"
