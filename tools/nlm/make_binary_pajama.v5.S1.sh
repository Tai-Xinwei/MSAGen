#!/usr/bin/bash
set -exuo pipefail

# check if environment variable SAS_TOKEN is set
# if [ -z "${SAS_TOKEN}" ]; then
#     echo "SAS_TOKEN is not set"
#     exit 1
# fi

BLOB_URL='https://sfmdataeastus2.blob.core.windows.net/nlm'

FILE_PATH_IN_BLOB_LIST=(
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part00.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part01.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part02.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part03.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part04.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part05.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part06.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part07.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part08.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part09.txt"
    "/sfmdataeastus2/nlm/renqianluo/data/SlimPajama/SlimPajama_train_sample_300B_part10.txt"
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
    FILE_PATH_IN_LOCAL=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/sfmdataeastus2\/nlm\/renqianluo//g')

    # check if local file exists, if not, use azcopy to download
    if [ ! -f "${FILE_PATH_IN_LOCAL}" ]; then
        FILE_RELATIVE_PATH=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/sfmdataeastus2\/nlm\///g')
        FILE_URL="${BLOB_URL}/${FILE_RELATIVE_PATH}"
        azcopy copy "${FILE_URL}" "${FILE_PATH_IN_LOCAL}"
    fi

    TARGET_FOLDER=$(printf "/data/SlimPajama/SlimPajama_%02d" ${ii})
    mkdir -p "${TARGET_FOLDER}"


    echo "processing train data"
    if [ ! -f "${TARGET_FOLDER}/train.npy" ]; then
        taskset -c $CPUID python -u tools/nlm/make_binary.S1.py \
        --input "${FILE_PATH_IN_LOCAL}" \
        --output ${TARGET_FOLDER}/train.npy \
        --tokenizer_path /data/Mixtral-8x7B-v0.1 \
        --num_workers 44 \
        --seq_len 8192 | tee ${TARGET_FOLDER}/data.log
    fi
done
echo "done"
