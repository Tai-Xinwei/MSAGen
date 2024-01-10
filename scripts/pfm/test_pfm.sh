#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=12
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=2
[ -z "${hidden_size}" ] && hidden_size=1024
[ -z "${ffn_size}" ] && ffn_size=2048
[ -z "${num_head}" ] && num_head=16
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=4
[ -z "${max_length}" ] && max_length=1024

[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${noise_mode}" ] && noise_mode=diff
[ -z "${mask_ratio}" ] && mask_ratio=0.15
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${total_num_steps}" ] && total_num_steps=1000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600
[ -z "${train_batch_size}" ] && train_batch_size=64
[ -z "${max_tokens}" ] && max_tokens=2048
[ -z "${val_batch_size}" ] && val_batch_size=67
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=2
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000000
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=3
[ -z "${seed}" ] && seed=42
[ -z "${checkpoint_dir}" ] && checkpoint_dir=""
[ -z "${which_set}" ] && which_set="valid"


[ -z "${mode_prob}" ] && mode_prob='1.0,0.0,0.0' # prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full
# cannot use DDP, since it does not implement valid_step...
[ -z "${strategy}" ] && strategy=Single

# [ -z "${data_path}" ] && data_path='/mnt/protein/48organism.lmdb/'
[ -z "${train_data_path}" ] && train_data_path='None'
[ -z "${valid_data_path}" ] && valid_data_path='None'
[ -z "${data_basepath}" ] && data_basepath="/mnta/yaosen/data/bfm_benchmark"
[ -z "${task_name}" ] && task_name="EnzymeCommission"
[ -z "${loadcheck_path}" ] && loadcheck_path="/mnta/yaosen/EnzymeCommission/checkpoint_E1.pt"
[ -z "${save_dir}" ] && save_dir="/mnta/yaosen/$task_name"
[ -z "${early_stopping}" ] && early_stopping=true
[ -z "${early_stopping_patience}" ] && early_stopping_patience=5
[ -z "${early_stopping_metric}" ] && early_stopping_metric='valid_loss'
[ -z "${early_stopping_mode}" ] && early_stopping_mode='min'
[ -z "${head_dropout}" ] && head_dropout=0.1
[ -z "${label_normalize}" ] && label_normalize=false

[ -z "${dataset_name}" ] && dataset_name="."
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0

if [[ "${early_stopping}" == "false" ]]
then
  early_stop_args=""
else
  early_stop_args="--early_stopping --early_stopping_patience $early_stopping_patience \
                   --early_stopping_metric $early_stopping_metric \
                   --early_stopping_mode $early_stopping_mode"
fi

if [[ "${label_normalize}" == "false" ]]
then
  label_normalize_args=""
else
  label_normalize_args="--label_normalize"
fi


python sfm/tasks/pfm/test_pfm_v2.py \
          --task_name $task_name \
          --data_basepath $data_basepath \
          --loadcheck_path $loadcheck_path \
          --encoder_attention_heads $num_head \
          --encoder_layers $layers \
          --encoder_ffn_embed_dim $ffn_size \
          --encoder_embed_dim $hidden_size \
          --droppath_prob $droppath_prob \
          --attn_dropout $attn_dropout \
          --num_3d_bias_kernel $num_3d_bias_kernel \
          --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
          --sandwich_ln \
          --dataset_names $dataset_name \
          --valid_data_path $valid_data_path \
          --train_data_path $train_data_path \
          --save_dir $save_dir \
          --seed $seed \
          --fp16 --ft \
          --mask_ratio $mask_ratio \
          --noise_scale $noise_scale \
          --num_pred_attn_layer $num_pred_attn_layer \
          --d_tilde $d_tilde \
          --strategy $strategy \
          --max_lr $max_lr \
          --mode_prob $mode_prob --noise_mode $noise_mode\
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --train_batch_size $train_batch_size --val_batch_size $val_batch_size \
          --max_tokens $max_tokens --max_length $max_length \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --save_epoch_interval $save_epoch_interval --total_num_epochs $epochs \
          --save_batch_interval $save_batch_interval --log_interval $log_interval \
          --head_dropout $head_dropout $early_stop_args $label_normalize_args --checkpoint_dir $checkpoint_dir --which_set $which_set
