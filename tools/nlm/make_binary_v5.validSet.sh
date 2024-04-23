#!/usr/bin/bash
set -exuo pipefail

# check if environment variable SAS_TOKEN is set
# if [ -z "${SAS_TOKEN}" ]; then
#     echo "SAS_TOKEN is not set"
#     exit 1
# fi

BLOB_URL='https://msralaphilly2.blob.core.windows.net/ml-la'

FILE_PATH_IN_BLOB_LIST=(
        /blob/v-zequnliu/mix_pretrain/valid.c4
        /blob/v-zequnliu/mix_pretrain/valid.pubmedabs
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/valid.pubchem.pureSMILES.randomRoot.10k
        /blob/lihe/scigpt/data/ur90/valid.uniref90.shuf.10k
        /blob/lihe/scigpt/data/dna/sequence/valid.DNASeq.20Btoken.10k
        /blob/lihe/scigpt/data/rna/20.0/valid.rnacentral.10k
)


declare -a TRAIN_FILES_LIST
L=${#FILE_PATH_IN_BLOB_LIST[@]}

echo "there are ${L} files to be processed"

CPUID="0-43"

for ((ii=0;ii<L;ii+=1)); do
    FILE_PATH_IN_BLOB=${FILE_PATH_IN_BLOB_LIST[${ii}]}
    echo $FILE_PATH_IN_BLOB

    # for FILE_PATH_IN_BLOB in "${FILE_PATH_IN_BLOB_LIST[@]}"; do
    FILE_PATH_IN_LOCAL=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/blob\//\/data\//g')

    # check if local file exists, if not, use azcopy to download
    if [ ! -f "${FILE_PATH_IN_LOCAL}" ]; then
        FILE_RELATIVE_PATH=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/blob\///g')
        FILE_URL="${BLOB_URL}/${FILE_RELATIVE_PATH}?${SAS_TOKEN}"
        # azcopy copy "${FILE_URL}" "${FILE_PATH_IN_LOCAL}"
    fi

    TRAIN_FILES_LIST+=("${FILE_PATH_IN_LOCAL}")
done

TRAIN_FILES=$(IFS=',' ; echo "${TRAIN_FILES_LIST[*]}")

TARGET_FOLDER="/data/processed/v5_validation"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing valid data"
if [ ! -f "${TARGET_FOLDER}/valid.npy" ]; then
        python -u tools/nlm/make_binary.py \
        --input "${TRAIN_FILES}" \
        --output ${TARGET_FOLDER}/valid.npy \
        --tokenizer_path /data/Mixtral-8x7B-v0.1 \
        --num_workers 8 \
        --seq_len 8192 | tee ${TARGET_FOLDER}/data.log
fi
echo "done"
