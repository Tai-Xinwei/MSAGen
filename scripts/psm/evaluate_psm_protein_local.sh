#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

num_sampling_time=10

# MODEL_CONFIG=PSM1B_DIT
# MODEL_CONFIG=PSM1B_exp3

MODEL_CONFIG=PSM3B_exp3
# MODEL_CONFIG=PSM3B_unify

# ckpt_folder_path=/data/peiran/blob/sfmdatawestus/psm/sfmexpresults/peiran/psmv1_mi300_edm_unify_v22_3b_stage1_5c_2/checkpoints
# ckpt_folder_path=/data/peiran/blob/sfmdatawestus/psm/sfmexpresults/peiran/psmv1_mi300_edm_exp3_v22_3b_ps_stage1_5c_3/checkpoints

global_step=global_step2500
ckpt_folder_path=/data/peiran/output/dit300m/

# global_step=global_step160000
CKPT_PATH=$ckpt_folder_path/$global_step/mp_rank_00_model_states.pt

SMPL_PATH=/home/peiranjin/output/psp/$global_step/prediction

# MODEL_CONFIG=PSM1B_unify
# ckpt_folder_path=/data/peiran/blob/sfmdatawestus/psm/sfmexpresults/shiyu/psm-checkpoints/psm-unified-20241228-0909
# global_step=checkpoint_E2_B97620
# CKPT_PATH=$ckpt_folder_path/checkpoint_E2_B97620.pt
# SMPL_PATH=/home/peiranjin/output/psp/$global_step/prediction

master_port=6667

DDP_TIMEOUT_MINUTES=3000 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port $master_port sfm/tasks/psm/pretrain_psm.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  max_length=2048 \
  mask_ratio=0.0 \
  data_path=/fastdata/peiran/psm \
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
  num_sampling_time=$num_sampling_time \
  loadcheck_path=$CKPT_PATH \
  sampled_structure_output_path=$SMPL_PATH \
  diffusion_mode=edm \
  use_memory_efficient_attention=false \
  psm_finetune_noise_mode=T \
  finetune_module=plddt_confidence_head \
  psm_sample_structure_in_finetune=True \
  psm_finetune_mode=True \

echo $CKPT_PATH

# ./tools/protein_evaluation/EvaluateProteinTest.py /fastdata/peiran/psm/ProteinTest/cameo-subset-casp14-and-casp15-combined.lmdb/ $SMPL_PATH $num_sampling_time $global_step
