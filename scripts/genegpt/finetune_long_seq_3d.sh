#!/bin/bash

# This is your argument
# data_path=$1
# kmer=$2
data_path=/home/v-zekunguo/zekun_data/gene/gue
kmer=6
echo "The provided kmer is: $kmer, data_path is $data_path"
# while ps -p "2994279" > /dev/null
# do
#     sleep 5m
# done

echo "Process is not running."
echo "Executing command..."

# sh scripts/run_dna1.sh 3 ; sh scripts/run_dna1.sh 4 ; sh scripts/run_dna1.sh 5 ; sh scripts/run_dna1.sh 6

for seed in 42
do

    python sfm/tasks/genegpt/finetune_long_seq_3d.py \
        --data_path  InstaDeepAI/genomics-long-range-benchmark \
        --task_name variant_effect_gene_expression \
        --sequence_length 8192 \
        --pretrained_ckpt_path /home/v-zekunguo/nlm/zekun/gene/checkpoints/1b6kmer4k_3d/global_step8000 \
        --model_config_path /home/v-zekunguo/nlm/zekun/gene/checkpoints/1b6kmer4k_3d/config/config.json \
        --kmer ${kmer} \
        --run_name 1bstep13k_long_seq_seed${seed} \
        --model_max_length 512 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --fp16 \
        --save_steps 500 \
        --output_dir /home/v-zekunguo/blob/v-zekunguo/gene/down/gue/1b/checkpoint \
        --evaluation_strategy steps \
        --eval_steps 500 \
        --warmup_steps 50 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --find_unused_parameters False

done
    # for data in prom_core_all prom_core_notata
    #     do
    #         python sfm/tasks/genegpt/finetune_gue.py \
    #             --model_name_or_path zhihan1996/DNA_bert_${kmer} \
    #             --data_path  ${data_path}/GUE/prom/$data \
    #             --kmer ${kmer} \
    #             --run_name 1b_${kmer}_prom_${data}_seed${seed} \
    #             --model_max_length 4096 \
    #             --per_device_train_batch_size 8 \
    #             --per_device_eval_batch_size 16 \
    #             --gradient_accumulation_steps 1 \
    #             --learning_rate 3e-5 \
    #             --num_train_epochs 4 \
    #             --fp16 \
    #             --save_steps 400 \
    #             --output_dir /home/v-zekunguo/blob/v-zekunguo/gene/down/gue/1b/checkpoint \
    #             --evaluation_strategy steps \
    #             --eval_steps 400 \
    #             --warmup_steps 50 \
    #             --logging_steps 100000 \
    #             --overwrite_output_dir True \
    #             --log_level info \
    #             --seed ${seed} \
    #             --find_unused_parameters False
    #     done
    # for data in prom_core_tata
    # do
    #     python sfm/tasks/genegpt/finetune_gue.py \
    #         --model_name_or_path zhihan1996/DNA_bert_${kmer} \
    #         --data_path  ${data_path}/GUE/prom/$data \
    #         --kmer ${kmer} \
    #         --run_name 1b_${kmer}_prom_${data}_seed${seed} \
    #         --model_max_length 80 \
    #         --per_device_train_batch_size 8 \
    #         --per_device_eval_batch_size 16 \
    #         --gradient_accumulation_steps 1 \
    #         --learning_rate 3e-5 \
    #         --num_train_epochs 10 \
    #         --fp16 \
    #         --save_steps 200 \
    #         --output_dir /home/v-zekunguo/blob/v-zekunguo/gene/down/gue/1b/checkpoint \
    #         --evaluation_strategy steps \
    #         --eval_steps 200 \
    #         --warmup_steps 50 \
    #         --logging_steps 100000 \
    #         --overwrite_output_dir True \
    #         --log_level info \
    #         --seed ${seed} \
    #         --find_unused_parameters False
    # done


    # for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    # do
    #     python sfm/tasks/genegpt/finetune_gue.py \
    #         --model_name_or_path zhihan1996/DNA_bert_${kmer} \
    #         --data_path  ${data_path}/GUE/EMP/$data \
    #         --kmer ${kmer} \
    #         --run_name 1b_${kmer}_EMP_${data}_seed${seed} \
    #         --model_max_length 4096 \
    #         --per_device_train_batch_size 8 \
    #         --per_device_eval_batch_size 8 \
    #         --gradient_accumulation_steps 2 \
    #         --learning_rate 3e-5 \
    #         --num_train_epochs 15 \
    #         --fp16 \
    #         --save_steps 200 \
    #         --output_dir /home/v-zekunguo/blob/v-zekunguo/gene/down/gue/1b/checkpoint \
    #         --evaluation_strategy steps \
    #         --eval_steps 200 \
    #         --warmup_steps 50 \
    #         --logging_steps 100000 \
    #         --overwrite_output_dir True \
    #         --log_level info \
    #         --seed ${seed} \
    #         --find_unused_parameters False
    # done
