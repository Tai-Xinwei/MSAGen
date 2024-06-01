

FILE_PATH_IN_BLOB_LIST=(
        "/data/processed_zekun/v5_processed_0_1/train.npy.bin,
        /data/processed_zekun/v5_processed_1_9/train.npy.bin,
        /data/processed_zekun/v5_processed_9_18/train.npy.bin,
        /data/processed_zekun/v5_processed_18_26/train.npy.bin,
        /data/processed_zekun/v5_processed_26_32/train.npy.bin,
        /data/processed_zekun/v5_processed_32_36/train.npy.bin,"

)
# FILE_PATH_IN_BLOB_LIST=(
#         "/data/processed_zekun/v5_processed_36_37/train.npy.bin"
# )

python \
tools/nlm/binary_to_lmdb.py \
"$FILE_PATH_IN_BLOB_LIST" \
/data/lmdb
