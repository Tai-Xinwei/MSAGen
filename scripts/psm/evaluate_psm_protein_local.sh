#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_CONFIG=PSM1B_V0
CKPT_PATH=/casp/sfm/sfmexpresults/shiyu/psm-checkpoints/pubchem-pm6-diffusion-molecule-protein-periodic-8xG8-fp32-ddp-unified-sampler-continued-fastpreprocess-20240725-1050/checkpoint_E0_B42500.pt
SMPL_PATH=/casp/sfm/sfmexpresults/jianwei/pubchem-pm6-diffusion-molecule-protein-periodic-8xG8-fp32-ddp-unified-sampler-continued-fastpreprocess-20240725-1050/checkpoint_E0_B42500-prediction

DDP_TIMEOUT_MINUTES=3000 torchrun --nproc_per_node gpu sfm/tasks/psm/pretrain_psm.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  max_length=2048 \
  mask_ratio=0.0 \
  data_path=/casp/sfm/psm \
  data_path_list=ProteinTest/cameo-subset-casp14-and-casp15-combined.lmdb \
  dataset_name_list=proteintest \
  dataset_split_raito=1.0 \
  use_unified_batch_sampler=false \
  val_batch_size=2 \
  val_batch_log_interval=1 \
  num_sampling_time=1 \
  loadcheck_path=$CKPT_PATH \
  sampled_structure_output_path=$SMPL_PATH \
