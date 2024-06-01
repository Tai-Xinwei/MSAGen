#!/usr/bin/bash
set -exuo pipefail

# check if environment variable SAS_TOKEN is set
# if [ -z "${SAS_TOKEN}" ]; then
#     echo "SAS_TOKEN is not set"
#     exit 1
# fi

BLOB_URL='https://msralaphilly2.blob.core.windows.net/ml-la'

FILE_PATH_IN_BLOB_LIST=(
       	/data/SlimPajama/300B_split/output_0.txt
        /data/SlimPajama/300B_split/output_1.txt
        /data/SlimPajama/300B_split/output_2.txt
        /data/SlimPajama/300B_split/output_3.txt
        /data/SlimPajama/300B_split/output_4.txt
        /data/SlimPajama/300B_split/output_5.txt
        /data/SlimPajama/300B_split/output_6.txt
        /data/SlimPajama/300B_split/output_7.txt
        /data/SlimPajama/300B_split/output_8.txt
        /data/SlimPajama/300B_split/output_9.txt
        /data/SlimPajama/300B_split/output_10.txt
        /data/SlimPajama/300B_split/output_11.txt
        /data/SlimPajama/300B_split/output_12.txt
        /data/SlimPajama/300B_split/output_13.txt
        /data/SlimPajama/300B_split/output_14.txt
        /data/SlimPajama/300B_split/output_15.txt
        /data/SlimPajama/300B_split/output_16.txt
        /data/SlimPajama/300B_split/output_17.txt
        /data/SlimPajama/300B_split/output_18.txt
        /data/SlimPajama/300B_split/output_19.txt
        /data/SlimPajama/300B_split/output_20.txt
        /data/SlimPajama/300B_split/output_21.txt
        /data/SlimPajama/300B_split/output_22.txt
        /data/SlimPajama/300B_split/output_23.txt
        /data/SlimPajama/300B_split/output_24.txt
        /data/SlimPajama/300B_split/output_25.txt
        /data/SlimPajama/300B_split/output_26.txt
        /data/SlimPajama/300B_split/output_27.txt
        /data/SlimPajama/300B_split/output_28.txt
        /data/SlimPajama/300B_split/output_29.txt
        /data/SlimPajama/300B_split/output_30.txt
        /data/SlimPajama/300B_split/output_31.txt
	/data/SlimPajama/train.patent.v2.txt
	/data/SlimPajama/material_doc_from_ziheng
	/data/SlimPajama/GPT4-rewrite-nc-cell-medrxiv.txt
	/data/SlimPajama/clinical_trial_intro.txt
    /data/SlimPajama/valid.patent.v2.txt


)


declare -a TRAIN_FILES_LIST
L=${#FILE_PATH_IN_BLOB_LIST[@]}
S=$1
E=$2
CPUID=$3

echo "there are ${L} files to be processed_zekun"
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

TARGET_FOLDER="/data/processed_zekun/v5_processed_${S}_${E}"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"
if [ ! -f "${TARGET_FOLDER}/train.npy" ]; then
        taskset -c $CPUID python -u tools/nlm/make_binary_test.py \
        --input "${TRAIN_FILES}" \
        --output ${TARGET_FOLDER}/train.npy \
        --tokenizer_path /home/yinxia/zekuntmp/SFM_framework/llama \
        --num_workers 44 \
        --seq_len 8192 | tee ${TARGET_FOLDER}/data.log
fi
echo "done"
