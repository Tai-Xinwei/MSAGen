#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_CONFIG=PSM3B_exp3
NUM_SAMPLING_TIME=10
GLOBAL_STEP=global_step100000
DATA_PATH=/casp/sfm/psm
WORK_PATH=/casp/sfm/sfmexpresults/peiran/psmv1_mi300_edm_exp3_v22_3b_ps_stage1_5c/checkpoints
CKPT_PATH=$WORK_PATH/$GLOBAL_STEP/mp_rank_00_model_states.pt
SMPL_PATH=$WORK_PATH/$GLOBAL_STEP/proteintest

DDP_TIMEOUT_MINUTES=3000 torchrun --nproc_per_node gpu sfm/tasks/psm/pretrain_psm.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  max_length=2048 \
  mask_ratio=0.0 \
  data_path=$DATA_PATH \
  data_path_list=ProteinTest/cameo-subset-casp14-and-casp15-combined.lmdb \
  dataset_name_list=proteintest \
  dataset_split_raito=1.0 \
  dataset_micro_batch_size=1 \
  use_unified_batch_sampler=true \
  val_batch_size=1 \
  val_batch_log_interval=1 \
  gradient_accumulation_steps=1 \
  diffusion_sampling=edm \
  num_timesteps_stepsize=-250 \
  num_sampling_time=$NUM_SAMPLING_TIME \
  loadcheck_path=$CKPT_PATH \
  sampled_structure_output_path=$SMPL_PATH \
  diffusion_mode=edm \
  use_memory_efficient_attention=false \

echo $CKPT_PATH
echo $SMPL_PATH

python tools/protein_evaluation/EvaluateProteinTest.py \
  $DATA_PATH/ProteinTest/cameo-subset-casp14-and-casp15-combined.lmdb \
  $SMPL_PATH \
  $NUM_SAMPLING_TIME \
  $GLOBAL_STEP \
