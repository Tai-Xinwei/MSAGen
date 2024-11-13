#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

num_sampling_time=10

# MODEL_CONFIG=PSM1B_DIT
MODEL_CONFIG=PSM1B_exp3
global_step=global_step50000
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_dit_v20_1b_stage1/checkpoints2
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp_v20_1b_stage1_nopdb/checkpoints
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp2_v21_1b_stage1_nocomplex/checkpoints
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp_v20_1b_stage1_pdbcomplex_2/checkpoints
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp_v20_1b_stage1_nopdb_stage2/checkpoints
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp3_v21_1b_stage1_all/checkpoints
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp3_v22_1b_stage1_ps_stage1_2/checkpoints
ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp3_v21_1b_stage1_ps_stage2_2/checkpoints

# global_step=global_step500
# ckpt_folder_path=/data/peiran/output/dit300m/

CKPT_PATH=$ckpt_folder_path/$global_step/mp_rank_00_model_states.pt
SMPL_PATH=/home/peiranjin/output/psp/$global_step/prediction

DDP_TIMEOUT_MINUTES=3000 torchrun --nproc_per_node gpu sfm/tasks/psm/pretrain_psm.py \
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

echo $CKPT_PATH

./tools/protein_evaluation/EvaluateProteinTest.py /fastdata/peiran/psm/ProteinTest/cameo-subset-casp14-and-casp15-combined.lmdb/ $SMPL_PATH $num_sampling_time $global_step
