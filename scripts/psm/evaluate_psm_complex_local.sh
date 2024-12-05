#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

num_sampling_time=10
MODEL_CONFIG=PSM1B_exp3

global_step=global_step35000
ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_edm_exp3_v22_1b_stage1_ps_stage2_h100_2/checkpoints

# global_step=global_step80000
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/kaiyuan/psm-dit/ft-edm-20241112-lr2e-5-bsz2-steps400000-warm25000-holo/

# global_step=global_step8848
# ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/kaiyuan/psm-dit/ft-edm-20241105-lr2e-5-bsz2-steps400000-warm25000-holo

global_step=global_step47304
ckpt_folder_path=/data/peiran/output/dit300m/

CKPT_PATH=$ckpt_folder_path/$global_step/mp_rank_00_model_states.pt
SMPL_PATH=/home/peiranjin/output/complex/$global_step/prediction

DDP_TIMEOUT_MINUTES=3000 torchrun --nproc_per_node gpu sfm/tasks/psm/pretrain_psm.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  complex_mode_prob=\"0.0,1.0,0.0\" \
  max_length=2048 \
  mask_ratio=0.0 \
  data_path=/fastdata/peiran/psm \
  data_path_list=ComplexTest/posebusters-428structures-20240828-c3302a23.removeLIGs.removeHs.lmdb \
  dataset_name_list=complextest \
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
  crop_radius=10000 \
  max_residue_num=20480 \
  ligand_crop_size=10000 \
  diffusion_mode=edm \
  use_memory_efficient_attention=false \
  # sample_ligand_only=true \

pocket_boundary=-1
result_path=$SMPL_PATH/../result.csv

python ./tools/protein_evaluation/EvaluateComplexAligned.py $SMPL_PATH /data/peiran/blob/sfmarca100/sfm/psm/PoseBusters/posebusters_benchmark_set $num_sampling_time $result_path $pocket_boundary

python ./tools/protein_evaluation/posebusters_stat.py $result_path $num_sampling_time
