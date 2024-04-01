#!/bin/bash

# Create the output directory
mkdir -p ./output

# Export the environment variables
export path=run.sh
export layers=12
export num_pred_attn_layer=2
export hidden_size=768
export ffn_size=3072
export num_head=32
export num_3d_bias_kernel=8
export atom_loss_coeff=1.0
export pos_loss_coeff=1.0
export sandwich_ln="true"
export dropout=0.0
export attn_dropout=0.1
export act_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.0
export noise_mode=diff
export noise_scale=0.2
export mask_ratio=0.5
export mode_prob=0.1,0.5,0.2,0.2
export d_tilde=1.0
export max_lr=1e-4
export strategy=DDP
export pipeline_model_parallel_size=0
export total_num_steps=2000000
export warmup_num_steps=1000
export train_batch_size=512 # 1024
export val_batch_size=512 # 1024
export max_tokens=4000
export max_length=128
export gradient_accumulation_steps=4
export lamb_pde_q=0
export lamb_pde_control=0.1
export diffmode=score
export wandb_group=tox_pde
export wandb_project=pdediffusion
export WANDB_RUN_NAME="withPDE_control"
export log_interval=100
export OMPI_COMM_WORLD_SIZE=1
export loadcheck_path=/blob/pfmexp/output/pfmdiff100M768_prob1522_m5_bs256_ddpmnoise_v1_pi_withpde_score/checkpoints
export data_path=/blob/data/afdb/48organism1m.lmdb/
export save_dir=/blob/pfmexp/output/pfmdiff100M768_prob1522_m5_bs256_ddpmnoise_v1_pi_withpde_score/checkpoints

# Create a temporary data directory
mkdir -p /tmp/data/pm6-86m-3d-filter

eval "$(conda shell.bash hook)"
conda activate sfm

python setup_cython.py build_ext --inplace

# Run training script
bash ./scripts/tox/pretrain_pdetox.sh
