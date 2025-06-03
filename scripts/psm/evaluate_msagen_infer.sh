#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_CONFIG=config_msagen_1B
NUM_SAMPLING_TIME=1
WORK_NAME=uniprot-all-1B-AR-1-32-avg-weightD-random-total2048-lr2e-5
WORK_PATH=/psm/xinwei/sfmexpresults/MSAGen_v2/$WORK_NAME
STEP_FLAG=global_step20000

DATA_PATH=../msadata
DATA_LMDB=casp15_length_lessthan128.lmdb

psm_validate_for_train_set=false
save_dir_base=./output/casp15_lessthan128/$WORK_NAME/$STEP_FLAG/
master_port=8891

for num in $(seq 1 1); do
  save_dir="${save_dir_base}/seed_${num}"

  echo "▶️ Running seed $num → save_dir: $save_dir"

  DDP_TIMEOUT_MINUTES=3000 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port $master_port sfm/tasks/psm/pretrain_msagen.py \
    --config-name=$MODEL_CONFIG \
    psm_validation_mode=true \
    sample_in_validation=true \
    mode_prob=\"0.0,1.0,0.0\" \
    complex_mode_prob=\"0.0,1.0,0.0\" \
    max_length=256 \
    mask_ratio=0.0 \
    data_path=$DATA_PATH \
    data_path_list=$DATA_LMDB \
    dataset_name_list=msageneration \
    dataset_micro_batch_size=1 \
    use_unified_batch_sampler=true \
    val_batch_size=1 \
    val_batch_log_interval=1 \
    gradient_accumulation_steps=1 \
    num_timesteps_stepsize=-1 \
    num_sampling_time=$NUM_SAMPLING_TIME \
    loadcheck_path=$WORK_PATH/$STEP_FLAG/mp_rank_00_model_states.pt \
    diffusion_mode=OADM \
    psm_validate_for_train_set=$psm_validate_for_train_set \
    cutoff=64 \
    random_select_msa=false \
    save_dir=$save_dir \
    keep_clean_num=2 \
    mode=1 \
    seed=$num \
    OADM_row_random=true

done

echo "✅ All 64 seeds finished."
