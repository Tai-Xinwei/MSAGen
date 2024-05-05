#!/bin/bash

# variables for distributed
export pipeline_model_parallel_size=0
export strategy=Zero1 # option:(Single, DDP, Zero1, Zero2, Zero3, Pipeline, ThreeD)
export launcher='openmpi'
export hostfile='/job/hostfile'
export MASTER_PORT=62447
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_SIZE=1


# Job specific variables
export lamb_ism=0.01
export lamb_pde_q=0
export lamb_pde_control=0.0
export WANDB_PROJECT="SFM_tox"
export WANDB_TEAM="large-scale-pde"
export wandb_group="pdm_dev"
mkdir -p /blob/pfmexp/output/pfmdiff300M_prob1261_m5_bs1024_ddpm_pair_G96/checkpoints
mkdir -p ./output
export path="run.sh"
export layers=24
export hidden_size=1024
export ffn_size=4096
export num_head=32
export sandwich_ln="true"
export dropout=0.0
export attn_dropout=0.1
export act_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.0
export noise_mode="diff"
export mask_ratio=0.5
export mode_prob="0.1,0.2,0.6,0.1"
export d_tilde=1.0
export max_lr=2e-4
export strategy="Zero1"
export pipeline_model_parallel_size=0
export total_num_steps=2000000
export warmup_num_steps=1000
export train_batch_size=16
export val_batch_size=16
export max_length=1024
export gradient_accumulation_steps=1
export log_interval=100
export loadcheck_path="/blob/pfmexp/output/pfmdiff300M_prob1261_m5_bs1024_ddpm_pair_G96_pde/checkpoints"
export data_path="/home/pisquare/data/AFDB50-plddt70.lmdb/"
export save_dir="/blob/pfmexp/output/pfmdiff300M_prob1261_m5_bs1024_ddpm_pair_G96_pde/checkpoints"

# Activate environment and run setup
eval "$(conda shell.bash hook)" && conda activate sfm
python setup_cython.py build_ext --inplace

# Run scripts
bash ./scripts/tox/pretrain_pdetox.sh
