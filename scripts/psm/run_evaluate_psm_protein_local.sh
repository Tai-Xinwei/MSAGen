#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copy variables from amulet yaml file
export NODES=8
export GPUS_PER_NODE=8
export WANDB_API_KEY="local-094f941ede8eda7a00c307f50595f054be5382f7"
export pbc_cutoff=20.0
export pbc_expanded_num_cell_per_direction=5
export pbc_expanded_token_cutoff=256
export pbc_multigraph_cutoff=5.0
export pbc_use_local_attention=False
export num_pred_attn_layer=4
export dataset_split_raito=0.4,0.1,0.4,0.1
export save_batch_interval=2500
export train_batch_size=1024
export val_batch_size=1024
export gradient_accumulation_steps=4
export val_batch_interval=0
export WANDB_RUN_NAME=psm-mol-pro-periodic-diff-relpos-noise1-relativepos-diffnoise10-${NODES}xG${GPUS_PER_NODE}-fp32-ddp-unified-sampler-fastpreprocess-20240606-2035
export total_num_steps=2000000
export warmup_num_steps=12000
export max_lr=2e-4
export diffusion_noise_std=10.0
export equivar_vec_init=ZERO_CENTERED_POS
export strategy=DDP
export fp16=False
export clean_sample_ratio=0.5
export mode_prob=0.1,0.2,0.7
export diff_init_lattice_size=10.0
export diffusion_sampling="ddpm"
export num_timesteps=5000
export ddpm_beta_start=1e-7
export ddpm_beta_end=2e-3
export ddpm_schedule=sigmoid
export equivar_use_linear_bias=True
export equivar_use_attention_bias=True
export data_path_list="PubChemQC-B3LYP-PM6,matter-sim-15M-merged,AFDB50-plddt70.lmdb,matter-sim-15M-force-filtered-merged"
export dataset_name_list="pm6,mattersim,afdb,mattersim"
export dataset_split_raito="0.4,0.1,0.4,0.1"
export dataset_micro_batch_size="32,8,4,8"
export use_unified_batch_sampler=True
export rescale_loss_with_std=True
export only_use_rotary_embedding_for_protein=True
export use_memory_efficient_attention=False

# variables for test
export max_length=2048
export psm_validation_mode=True
export data_path="/casp/jianwzhu/workspace/SFM_Evaluation/run_sfm/sfmblob/psm/cameo-subset-casp14-and-casp15-combined.lmdb"
export loadcheck_path="/casp/jianwzhu/workspace/SFM_Evaluation/run_sfm/sfmblob/psm-checkpoints/pubchem-pm6-diffusion-molecule-protein-periodic-8xG8-fp32-ddp-unified-sampler-continued-fastpreprocess-20240607-2159/checkpoint_E2_B29070.pt"
export save_dir="/casp/jianwzhu/workspace/SFM_Evaluation/run_sfm/sfmblob/psm-checkpoints/pubchem-pm6-diffusion-molecule-protein-periodic-8xG8-fp32-ddp-unified-sampler-continued-fastpreprocess-20240607-2159/checkpoint_E2_B29070-prediction"
HERE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
inpsh="$HERE/pretrain_psm.sh"
outsh="$HERE/evaluate_psm_protein.sh"
sed 's/pretrain_psm.py/evaluate_psm_protein.py/' $inpsh > $outsh
bash $outsh && rm $outsh
