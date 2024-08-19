#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# MODEL_CONFIG=PSM1B_V0
# CKPT_PATH=/casp/sfm/sfmexpresults/shiyu/psm-checkpoints/pubchem-pm6-diffusion-molecule-protein-periodic-8xG8-fp32-ddp-unified-sampler-continued-fastpreprocess-20240725-1050/checkpoint_E0_B42500.pt
# SMPL_PATH=/casp/sfm/sfmexpresults/jianwei/pubchem-pm6-diffusion-molecule-protein-periodic-8xG8-fp32-ddp-unified-sampler-continued-fastpreprocess-20240725-1050/checkpoint_E0_B42500-prediction
# MODEL_CONFIG=PSM300M_DIT
# CKPT_PATH=/casp/sfm/sfmexpresults/peiran/psmv1_dit_v13_300m/checkpoints/global_step80000/mp_rank_00_model_states.pt
# SMPL_PATH=/casp/sfm/sfmexpresults/jianwei/psmv1_dit_v13_300m/checkpoints/global_step80000/prediction
MODEL_CONFIG=PSM1B_DIT
# CKPT_PATH=/casp/sfm/sfmexpresults/peiran/psmv1_dit_v13_1b/checkpoints/global_step75000/mp_rank_00_model_states.pt
# SMPL_PATH=/casp/sfm/sfmexpresults/jianwei/psmv1_dit_v13_1b/checkpoints/global_step75000/prediction

num_sampling_time=5
global_step=global_step5000
ckpt_folder_path=/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_dit_v13_1b/checkpoints_2
# global_step=global_step500
# ckpt_folder_path=/data/peiran/output

CKPT_PATH=$ckpt_folder_path/$global_step/mp_rank_00_model_states.pt
SMPL_PATH=/home/peiranjin/output/$global_step/prediction


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
  diffusion_sampling=dpm \
  num_timesteps_stepsize=-250 \
  num_sampling_time=$num_sampling_time \
  loadcheck_path=$CKPT_PATH \
  sampled_structure_output_path=$SMPL_PATH \

./tools/protein_evaluation/EvaluateProteinTest.py /fastdata/peiran/psm/ProteinTest/cameo-subset-casp14-and-casp15-combined.lmdb/ $SMPL_PATH $num_sampling_time
