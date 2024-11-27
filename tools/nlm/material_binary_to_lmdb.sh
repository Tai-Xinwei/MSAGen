

FILE_PATH_IN_BLOB_LIST=(
    "/data/SlimPajama/SlimPajama_1_2/train.npy.bin"
)

python tools/nlm/binary_to_lmdb.py \
"$FILE_PATH_IN_BLOB_LIST" \
/data/SlimPajama/SlimPajama_1_2/
