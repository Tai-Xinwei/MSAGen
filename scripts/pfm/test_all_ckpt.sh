#!/bin/bash

[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${down_stream_set}" ] && down_stream_set='valid'
[ -z "${run}" ] && run="0103-afternoon-evening"
[ -z "${seed}" ] && seed=21
[ -z "${base_path}" ] && base_path="/blob/shufxi/pfmexp/output/finetune.epoch100"

folder=$base_path/finetune-${task_name}_lr${max_lr}_seed${seed}_${run}

output_csv="$folder/scores.csv"
if [ ! -f "$output_csv" ]; then
    echo "checkpoint,score_dict" > "$output_csv"
fi

for ckpt in $folder/checkpoint*; do
    export loadcheck_path=$ckpt
    echo "Load checkpoint: $loadcheck_path"
    prog_output=$(bash scripts/pfm/test_pfm_bpe.sh 2>&1)

    checkpoint_path=$(echo "$prog_output" | grep -oP '(?<=Checkpoint: ).*(?=; Test results)')
    echo "Checkpoint: $checkpoint_path"

    scores=$(echo "$prog_output" | grep -oP '(?<=Test results: ).*')
    echo "Scores: $scores"

    echo "$checkpoint_path,\"$scores\"" >> "$output_csv"
done

cat "$output_csv"
