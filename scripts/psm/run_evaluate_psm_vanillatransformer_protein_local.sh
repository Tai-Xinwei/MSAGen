#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copy variables from amulet yaml file
export WANDB_TEAM=ai4s-sfm
export wandb_group=psm_VT
export wandb_project=psm_VT
export WANDB_API_KEY="local-094f941ede8eda7a00c307f50595f054be5382f7"
export layers=32
export hidden_size=1536
export ffn_size=6144
export num_head=32
export atom_loss_coeff=1.0
export pos_loss_coeff=1.0
export sandwich_ln="true"
export dropout=0.0
export attn_dropout=0.1
export act_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.0
export mask_ratio=0.5
export d_tilde=1.0
export max_lr=1.5e-4
export strategy=Zero1
export pipeline_model_parallel_size=0
export total_num_steps=500000
export warmup_num_steps=20000
export train_batch_size=1024
export val_batch_size=1024
export max_tokens=16000
export max_length=512
export gradient_accumulation_steps=8
export log_interval=100
export data_path=/psm/data/
export data_path_list='PubChemQC-B3LYP-PM6,matter-sim-15M-force-filtered-merged,AFDB70-plddt70.lmdb,matter-sim-15M-merged,ur50_23_bpe_pack1536.lmdb'
export dataset_name_list='pm6,mattersim,afdb,mattersim,ur50'
export dataset_split_raito='0.2,0.1,0.5,0.1,0.1'
export dataset_micro_batch_size='64,8,8,8,2'
export mode_prob='0.4,0.4,0.2'
export fp16=False
export clean_sample_ratio=0.2
export energy_loss_ratio=0.001
export force_loss_ratio=0.0
export use_unified_batch_sampler=True
export use_dali_pipeline=False
export save_batch_interval=10000
export only_use_rotary_embedding_for_protein=True
export loadcheck_path=/sfm/sfmexpresults/peiran/psmv1_vt_v10_1b/checkpoints
export save_dir=/sfm/sfmexpresults/peiran/psmv1_vt_v10_1b/checkpoints

# variables for test
export max_length=2048
export psm_validation_mode=True
export data_path="/casp/sfm/psm/cameo-subset-casp14-and-casp15-combined.lmdb"
export loadcheck_path="/casp/sfm/sfmexpresults/peiran/psmv1_vt_v10_1b/checkpoints/global_step103463/mp_rank_00_model_states.pt"
export save_dir="/casp/sfm/sfmexpresults/jianwei/psmv1_vt_v10_1b/checkpoints/global_step103463/prediction"
HERE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
inpsh="$HERE/pretrain_psm_vanillatransformer.sh"
outsh="$HERE/evaluate_psm_vanillatransformer_protein.sh"
cp $inpsh $outsh
sed -i 's/pretrain_psm.py/evaluate_psm_protein.py/' $outsh
sed -i 's/seed=12345/seed=12345 max_model_num=1/' $outsh
mkdir -p $save_dir
bash $outsh && rm $outsh
