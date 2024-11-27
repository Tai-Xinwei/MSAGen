[ -z "${output_dir}" ] && output_dir="/home/v-zekunguo/logs/1b/base_table"

mkdir -p $output_dir/all
for file_name in "$output_dir/"*; do
    base_name=$(basename "$file_name")

    if [ "$base_name" != "all" ]; then
        python tools/nlm/x_merge.py \
        --input_glob "$output_dir/$base_name/part*.pkl" \
        --output_file "$output_dir/all/$base_name.response.pkl"
    fi
done
