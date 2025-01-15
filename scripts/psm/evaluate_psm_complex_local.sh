#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CKPT_PATH=$ckpt_folder_path/$global_step/mp_rank_00_model_states.pt
SMPL_PATH=/home/peiranjin/output/complex/$global_step/prediction
MODEL_CONFIG=PSM3B_exp3
NUM_SAMPLING_TIME=50
GLOBAL_STEP=global_step160000
DATA_PATH=/fastdata/peiran/psm
WORK_PATH=/data/peiran/blob/sfmdatawestus/psm/sfmexpresults/peiran/psmv1_mi300_edm_exp3_v22_3b_ps_stage1_5c/checkpoints
CKPT_PATH=$WORK_PATH/$GLOBAL_STEP/mp_rank_00_model_states.pt
SMPL_PATH=$WORK_PATH/$GLOBAL_STEP/posebusters

master_port=6667

DDP_TIMEOUT_MINUTES=3000 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port $master_port sfm/tasks/psm/pretrain_psm.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  complex_mode_prob=\"0.0,1.0,0.0\" \
  max_length=2048 \
  mask_ratio=0.0 \
  data_path=$DATA_PATH \
  data_path_list=ComplexTest/posebusters-428structures-20240828-c3302a23.removeLIGs.removeHs.lmdb \
  dataset_name_list=complextest \
  dataset_split_raito=1.0 \
  dataset_micro_batch_size=1 \
  use_unified_batch_sampler=true \
  val_batch_size=1 \
  val_batch_log_interval=1 \
  gradient_accumulation_steps=1 \
  diffusion_sampling=dpm_edm \
  num_timesteps_stepsize=-250 \
  num_sampling_time=$NUM_SAMPLING_TIME \
  loadcheck_path=$CKPT_PATH \
  sampled_structure_output_path=$SMPL_PATH \
  crop_radius=10000 \
  max_residue_num=20480 \
  ligand_crop_size=10000 \
  diffusion_mode=edm \
  use_memory_efficient_attention=false \
  # sample_ligand_only=true \

echo $CKPT_PATH
echo $SMPL_PATH

python tools/protein_evaluation/EvaluateComplexAligned.py \
  $SMPL_PATH \
  /casp/sfm/psm/ComplexTest/posebusters_benchmark_set \
  $NUM_SAMPLING_TIME \
  $SMPL_PATH/../result.csv \
  -1

python tools/protein_evaluation/posebusters_stat.py \
  $SMPL_PATH/../result.csv \
  $NUM_SAMPLING_TIME
