#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copy variables from amulet yaml file
export WANDB_TEAM=ai4s-sfm
export wandb_group=psm_VT
export wandb_project=psm_VT
# export WANDB_API_KEY="local-094f941ede8eda7a00c307f50595f054be5382f7"
export path=run.sh
export num_pred_attn_layer=4
export layers=22
export hidden_size=1024
export ffn_size=4096
export num_head=32
export atom_loss_coeff=1.0
export pos_loss_coeff=1.0
export sandwich_ln="true"
export dropout=0.0
export attn_dropout=0.1
export act_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.0
export mask_ratio=0.0
export d_tilde=1.0
export max_lr=2.0e-4
export strategy=Zero1
export pipeline_model_parallel_size=0
export total_num_steps=500000
export warmup_num_steps=10000
export train_batch_size=1024
export val_batch_size=1024
export max_tokens=16000
export max_length=512
export gradient_accumulation_steps=4
export log_interval=100
export data_path=/home/v-zhezhan/preprocessed
export data_path_list='pdb'
export dataset_name_list='pm6'
export dataset_split_raito='1.0'
export dataset_micro_batch_size='128'
export fp16=False

# variables for test
export max_length=512
export psm_validation_mode=True
export data_path="/home/v-zhezhan/preprocessed"
export loadcheck_path="/home/v-zhezhan/mp_rank_00_model_states.pt"
export save_dir="/home/v-zhezhan/prediction"
#export data_path="/fastdata/peiran/psm/preprocessed.large"
#export loadcheck_path="/data/peiran/output/global_step130000/mp_rank_00_model_states.pt"
#export save_dir="/home/peiranjin/output/"
HERE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
inpsh="$HERE/finetune_psm_vanillatransformer_complex.sh"
outsh="$HERE/evaluate_psm_vanillatransformer_complex.sh"
sed 's/finetune_psm_complex.py/evaluate_psm_complex.py/' $inpsh > $outsh
bash $outsh && rm $outsh
