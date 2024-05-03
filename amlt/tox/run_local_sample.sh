#!/bin/bash

export path=run.sh

# variables for distributed
export pipeline_model_parallel_size=0
export strategy=Zero1 # option:(Single, DDP, Zero1, Zero2, Zero3, Pipeline, ThreeD)
export launcher='openmpi'
export hostfile='/job/hostfile'
export MASTER_PORT=62447
export MASTER_ADDR=127.0.0.1
export OMPI_COMM_WORLD_SIZE=1

# variables for optimizer
export gradient_accumulation_steps=1
export max_lr=1e-4
export weight_decay=0.0

# variables for batch size
export max_tokens=4000
export train_batch_size=256
export val_batch_size=10

# variables for log
export log_interval=20

# variables for wandb
export WANDB_GROUP="pde_q_sample"
export WANDB_TEAM="junzhe_personal"
export WANDB_PROJECT="pde_sample_angle"
export WANDB_RUN_NAME="sample_angle_test"
export wandb_group=${WANDB_GROUP}
export wandb_team=${WANDB_TEAM}
export wandb_project=${WANDB_PROJECT}
export wandb_run_name=${WANDB_RUN_NAME}

# varibales for training
export epochs=1000
export total_num_steps=2000000
export warmup_num_steps=1000
export save_epoch_interval=1
export save_batch_interval=10000000
# export data_path="/blob/data/afdb/48organisms-fullatom.lmdb/"
export data_path="/fastdata/peiran/tox/48organisms-fullatom.lmdb/"

export dataset_name="."
export save_dir="~/SFM_framework/output/sample_result"
export loadcheck_path="/blob/pfmexp/output/junzhe/checkpoints/pde_q_running_loss/pde_q_sample/epsilon_model_without_pde_v3/global_step147499/mp_rank_00_model_states.pt"

# variables for model
export layers=12
export hidden_size=768
export ffn_size=3072
export num_head=32
export num_pred_attn_layer=2
export atom_loss_coeff=1.0
export pos_loss_coeff=1.0
export max_length=128
export dropout=0.0
export attn_dropout=0.1
export act_dropout=0.1
export sandwich_ln=true
export droppath_prob=0.0
export noise_scale=0.2
export noise_mode=diff
export lamb_pde_q=0
export lamb_pde_control=0.01
export diffmode="epsilon"
# export seq_masking_method=continuousMask
export seq_masking_method=transformerM
export mask_ratio=0.0
export d_tilde=1.0
export mode_prob='0.0,0.0,1.0,0.0'
export add_3d=true
export no_2d=false
export num_3d_bias_kernel=8

# sample args
export ode_mode=false
export num_timesteps=1000

# Create the output directory
mkdir -p ${save_dir}

eval "$(conda shell.bash hook)"
conda activate sfm

python setup_cython.py build_ext --inplace

# Run training script
bash ./scripts/tox/sample_tox.sh
