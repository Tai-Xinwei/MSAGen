#!/usr/bin/bash
set -exuo pipefail

# check if environment variable SAS_TOKEN is set
# if [ -z "${SAS_TOKEN}" ]; then
#     echo "SAS_TOKEN is not set"
#     exit 1
# fi

BLOB_URL='https://msralaphilly2.blob.core.windows.net/ml-la'

FILE_PATH_IN_BLOB_LIST=(
        /home/v-zekunguo/data/chuancao/data
)


declare -a TRAIN_FILES_LIST
L=${#FILE_PATH_IN_BLOB_LIST[@]}
# S=$1
# E=$2
CPUID=$1

echo "there are ${L} files to be processed"
# echo "processing ${S} to ${E}"


# for ((ii=0;ii<L;ii+=1)); do
#     FILE_PATH_IN_BLOB=${FILE_PATH_IN_BLOB_LIST[${ii}]}
#     echo $FILE_PATH_IN_BLOB

#     # for FILE_PATH_IN_BLOB in "${FILE_PATH_IN_BLOB_LIST[@]}"; do
#     FILE_PATH_IN_LOCAL=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/blob\//\/data\//g')

#     # check if local file exists, if not, use azcopy to download
#     if [ ! -f "${FILE_PATH_IN_LOCAL}" ]; then
#         FILE_RELATIVE_PATH=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/blob\///g')
#         FILE_URL="${BLOB_URL}/${FILE_RELATIVE_PATH}?${SAS_TOKEN}"
#         # azcopy copy "${FILE_URL}" "${FILE_PATH_IN_LOCAL}"
#     fi

#     TRAIN_FILES_LIST+=("${FILE_PATH_IN_LOCAL}")
# done

TRAIN_FILES=$(IFS=',' ; echo "${TRAIN_FILES_LIST[*]}")

TARGET_FOLDER="/data/gene/lmdb"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"

taskset -c $CPUID python -u tools/gene_process/gene_fasta_to_lmdb.py \
--input "/data/gene/genome_selected" \
--output ${TARGET_FOLDER} \
--num_workers 32 \
--seq_len 16000 | tee ${TARGET_FOLDER}/data.log

echo "done"
