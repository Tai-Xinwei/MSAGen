#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_CONFIG=config_msagen_1B
NUM_SAMPLING_TIME=1

WORK_NAME=uniprot-easyrank-1B-112244-avg-total512-lr2e-5

WORK_PATH=/psm/xinwei/sfmexpresults/MSAGen_v2/$WORK_NAME
# WORK_PATH=/psm/sfmexpresults/xinwei/MSAGen/MSAGen_1000_2_to_2_change_ce_to_L1_loss_enlargediff5xbutnogap

STEP_FLAG=global_step10000

DATA_PATH=../msadata
DATA_LMDB=msas-uniprot-easyrank-subset.lmdb
# DATA_LMDB=posebusters-428structures-20250221-670e6562.removeLIGs.removeHs.lmdb

# if [[ $DATA_LMDB == *"proteintest"* ]]; then
#   SCRIPT=tools/protein_evaluation/EvaluateProteinTest.py
#   SMPL_PATH="$WORK_PATH/$STEP_FLAG/proteintest"
# elif [[ $DATA_LMDB == *"posebusters"* ]]; then
#   SCRIPT=tools/protein_evaluation/EvaluatePoseBusters.py
#   SMPL_PATH="$WORK_PATH/$STEP_FLAG/posebusters"
# else
#   echo "Dataset must be proteintest or posebusters"
#   exit 1
# fi
psm_validate_for_train_set=false

if [ "$psm_validate_for_train_set" = true ]; then
    save_dir=./output/$WORK_NAME/$STEP_FLAG/train_random_penalty0_prob_2-2
else
    save_dir=./output/$WORK_NAME/$STEP_FLAG/valid_random_penalty0_prob_2-2
fi
# save_dir=./output/msas-uniprot-easyrank-subset/
master_port=7796

DDP_TIMEOUT_MINUTES=3000 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port $master_port sfm/tasks/psm/pretrain_msagen.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  complex_mode_prob=\"0.0,1.0,0.0\" \
  max_length=384 \
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
  cutoff=4 \
  random_select_msa=false \
  save_dir=$save_dir \
  keep_clean_num=2 \
  mode=2 \
  OADM_row_random=true \
  # sample_ligand_only=true \

echo $CKPT_PATH


# python $SCRIPT $DATA_PATH/ComplexTest/$DATA_LMDB $SMPL_PATH
