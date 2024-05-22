

FILE_PATH_IN_BLOB_LIST=(
        "/home/v-zekunguo/zekun_data/scidata/valid.npy,
        /home/v-zekunguo/zekun_data/scidata/valid1.npy"

)


python \
tools/nlm/binary_to_lmdb.py \
"$FILE_PATH_IN_BLOB_LIST" \
/home/v-zekunguo/zekun_data/scidata/output
