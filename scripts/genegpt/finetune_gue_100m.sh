#!/bin/bash

# This is your argument
# data_path=$1
# kmer=$2
data_path=/home/v-zekunguo/zekun_data/gene/gue
kmer=6
echo "The provided kmer is: $kmer, data_path is $data_path"

# sh scripts/run_dna1.sh 3 ; sh scripts/run_dna1.sh 4 ; sh scripts/run_dna1.sh 5 ; sh scripts/run_dna1.sh 6

for seed in 42
do
    for data in H3
    do
        python sfm/tasks/genegpt/finetune_gue_100m.py \
            --model_name_or_path zhihan1996/DNA_bert_${kmer} \
            --data_path  ${data_path}/GUE/EMP/$data \
            --kmer ${kmer} \
            --run_name DNABERT1_${kmer}_EMP_${data}_seed${seed} \
            --model_max_length 4096 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --learning_rate 3e-5 \
            --num_train_epochs 15 \
            --fp16 \
            --save_steps 200 \
            --output_dir /home/v-zekunguo/blob/v-zekunguo/gene/down/gue/100m/checkpoint \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False
    done
done
