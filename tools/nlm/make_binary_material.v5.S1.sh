#!/usr/bin/bash
set -exuo pipefail

# check if environment variable SAS_TOKEN is set
# if [ -z "${SAS_TOKEN}" ]; then
#     echo "SAS_TOKEN is not set"
#     exit 1
# fi

BLOB_URL='https://msralaphilly2.blob.core.windows.net/ml-la'

FILE_PATH_IN_BLOB_LIST=(
    /msralaphilly2/ml-la/yinxia/wu2/shared/SFM/material/structures2/train_cf_sg.txt
    /msralaphilly2/ml-la/yinxia/wu2/shared/SFM/material/structures2/train_cf.txt
    /msralaphilly2/ml-la/yinxia/wu2/shared/SFM/material/structures2/train_fcf_sg.txt
    /msralaphilly2/ml-la/yinxia/wu2/shared/SFM/material/text_and_material_cf_sg.txt
    /msralaphilly2/ml-la/yinxia/wu2/shared/SFM/material/text_and_material_fcf_sg.txt
)


declare -a TRAIN_FILES_LIST
L=${#FILE_PATH_IN_BLOB_LIST[@]}
S=$1
E=$2
CPUID=$3

echo "there are ${L} files to be processed"
echo "processing ${S} to ${E}"


for ((ii=S;ii<E;ii+=1)); do
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

#TARGET_FOLDER="/data/processed/v5_processed_${S}_${E}"
TARGET_FOLDER="/sfmdataeastus2/nlm/SFMData/pretrain/20240724/train_split_lmdb/material.lmdb"
TARGET_FOLDER="/sfmdataeastus2/nlm/SFMData/pretrain/20240724/train_split_lmdb/text_and_material.lmdb"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"
if [ ! -f "${TARGET_FOLDER}/train.npy" ]; then
        taskset -c $CPUID python -u tools/nlm/make_binary.S1.py \
        --input "${TRAIN_FILES}" \
        --output ${TARGET_FOLDER}/train.npy \
        --tokenizer_path /sfmdataeastus2/nlm/Mixtral-8x7B-v0.1 \
        --num_workers 44 \
        --seq_len 8192 | tee ${TARGET_FOLDER}/data.log
fi
echo "done"
