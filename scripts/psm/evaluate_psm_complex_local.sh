#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_CONFIG=PSM3B_exp3
NUM_SAMPLING_TIME=5

WORK_PATH=/caspprem/jianwzhu/sfm/sfmexpresults/peiran/psmv1_mi300_edm_exp3_v22_3b_ps_stage1_5c_2/checkpoints
STEP_FLAG=global_step142500

DATA_PATH=/caspprem/jianwzhu/sfm/psm
DATA_LMDB=proteintest-casp14-cameo-casp15.20250221_76b5be8e.lmdb
DATA_LMDB=posebusters-428structures-20250221-670e6562.removeLIGs.removeHs.lmdb

if [[ $DATA_LMDB == *"proteintest"* ]]; then
  SCRIPT=tools/protein_evaluation/EvaluateProteinTest.py
  SMPL_PATH="$WORK_PATH/$STEP_FLAG/proteintest"
elif [[ $DATA_LMDB == *"posebusters"* ]]; then
  SCRIPT=tools/protein_evaluation/EvaluatePoseBusters.py
  SMPL_PATH="$WORK_PATH/$STEP_FLAG/posebusters"
else
  echo "Dataset must be proteintest or posebusters"
  exit 1
fi

master_port=6667

DDP_TIMEOUT_MINUTES=3000 CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port $master_port sfm/tasks/psm/pretrain_psm.py \
  --config-name=$MODEL_CONFIG \
  psm_validation_mode=true \
  sample_in_validation=true \
  mode_prob=\"0.0,1.0,0.0\" \
  complex_mode_prob=\"0.0,1.0,0.0\" \
  max_length=2048 \
  mask_ratio=0.0 \
  data_path=$DATA_PATH \
  data_path_list=ComplexTest/$DATA_LMDB \
  dataset_name_list=complextest \
  dataset_split_raito=1.0 \
  dataset_micro_batch_size=1 \
  use_unified_batch_sampler=true \
  val_batch_size=1 \
  val_batch_log_interval=1 \
  gradient_accumulation_steps=1 \
  diffusion_sampling=edm \
  num_timesteps_stepsize=-250 \
  num_sampling_time=$NUM_SAMPLING_TIME \
  loadcheck_path=$WORK_PATH/$STEP_FLAG/mp_rank_00_model_states.pt \
  sampled_structure_output_path=$SMPL_PATH \
  crop_radius=10000 \
  max_residue_num=20480 \
  ligand_crop_size=10000 \
  diffusion_mode=edm \
  use_memory_efficient_attention=false \
  psm_finetune_noise_mode=T \
  finetune_module=plddt_confidence_head \
  psm_sample_structure_in_finetune=True \
  psm_finetune_mode=True \
  # sample_ligand_only=true \

echo $CKPT_PATH
echo $SMPL_PATH

python $SCRIPT $DATA_PATH/ComplexTest/$DATA_LMDB $SMPL_PATH
