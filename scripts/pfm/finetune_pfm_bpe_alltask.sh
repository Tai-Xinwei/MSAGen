#!/bin/bash

export layers=12
export num_pred_attn_layer=2
export hidden_size=1024
export ffn_size=2048
export num_head=16
export atom_loss_coeff=1.0
export pos_loss_coeff=1.0
export sandwich_ln="true"
export dropout=0.1
export attn_dropout=0.1
export act_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.0
export max_num_aa=1024
export noise_mode=diff
export noise_scale=0.2
export mask_ratio=0.2
export mode_prob=1.0,0.0,0.0
export d_tilde=1.0
export strategy=DDP
export pipeline_model_parallel_size=0
export train_batch_size=128
export val_batch_size=128
export max_tokens=6400
export max_length=2048
export gradient_accumulation_steps=1
export log_interval=10
export epochs=100
export WANDB_PROJECT=pfm_finetune
export train_data_path="None"
export valid_data_path="None"
export data_basepath="/pfm/data/bfm_benchmark"
export loadcheck_path=/blob/shufxi/pfm/bpe/tiny/mask0.15_rd_lr3e-4_bsz2k_aa/checkpoint_E59.pt
export early_stopping=true
export early_stopping_patience=100
export early_stopping_metric='valid_loss'
export early_stopping_mode='min'
export head_dropout=0.1
export spm_model_path='/blob/shufxi/data/biofm/ur50bpe/ur50bpe.model'


ls /blob/shufxi/data/biofm/ur50bpe

for task in 'solubility' 'EnzymeCommission' 'GeneOntology_mf' 'GeneOntology_bp' 'GeneOntology_cc' 'beta_lactamase' 'fluorescence'  'stability' 'subcellular_localization' 'subcellular_localization_2' 'remote_homology_fold' 'human_ppi' 'yeast_ppi' 'ppi_affinity'
do
    export task_name=$task
    export WANDB_RUN_NAME="finetune-${task_name}_lr${max_lr}_seed${seed}_${run}"
    export save_dir=/blob/shufxi/pfmexp/output/finetune.epoch100/finetune-${task_name}_lr${max_lr}_seed${seed}_${run}
    mkdir -p "$save_dir"
    echo "=============================="
    echo "Task: $task_name"
    echo "=============================="
    bash ./scripts/pfm/finetune_pfm_bpe.sh
done
