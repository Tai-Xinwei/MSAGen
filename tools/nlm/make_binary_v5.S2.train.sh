input_str=$(ls /data/processed/v5_processed_*/*bin | paste -sd, -)

python \
make_binary.S2.py \
"$input_str" \
/data/processed/v5_train/train.npy
