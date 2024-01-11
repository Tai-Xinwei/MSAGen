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
[ -z "${model_type}" ] && model_type='pfm_mlm_tiny'

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
[ -z "${val_batch_size}" ] && val_batch_size=64
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=2
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000000
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=3
[ -z "${seed}" ] && seed=42


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

[ -z "${dataset_name}" ] && dataset_name="."
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0

[ -z "${wandb_group}" ] && wandb_group=tinyBFM-finetune
[ -z "${wandb_team}" ] && wandb_team=icuppjin
[ -z "${wandb_project}" ] && wandb_project=ds_mfmpre

[ -z "${down_stream_set}" ] && down_stream_set='test'
[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
# [ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=-1

# wandb login --relogin 5d03b7a46d10f86ff45c4aedc570660a523edc0b
# export WANDB_API_KEY=5d03b7a46d10f86ff45c4aedc570660a523edc0b

echo -e "\n\n"
echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
echo "n_gpu: ${n_gpu}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "LOCAL_RANK : ${LOCAL_RANK}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"
echo "OMPI_COMM_WORLD_LOCAL_RANK: ${OMPI_COMM_WORLD_LOCAL_RANK}"

# echo "AZUREML_EXPERIMENT_ID: ${AZUREML_EXPERIMENT_ID}"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "n_layers: ${layers}"
echo "num_pred_attn_layer: ${num_pred_attn_layer}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "d_tilde: ${d_tilde}"
echo "sandwich_ln: ${sandwich_ln}"
echo "max_lr: ${max_lr}"
echo "total_num_steps: ${total_num_steps}"
echo "warmup_num_steps: ${warmup_num_steps}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "droppath_prob: ${droppath_prob}"
echo "atom_loss_coeff: ${atom_loss_coeff}"
echo "pos_loss_coeff: ${pos_loss_coeff}"
echo "no_2d: ${no_2d}"
echo "add_3d: ${add_3d}"
echo "data_path: ${data_path}"
echo "output_path: ${output_path}"
echo "dataset_name: ${dataset_name}"
echo "noise_scale: ${noise_scale}"
echo "mask_ratio: ${mask_ratio}"
echo "mode_prob: ${mode_prob}"
echo "noise_mode: ${noise_mode}"
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"



python3 sfm/tasks/pfm/test_pfm.py \
--base_model pfm_bpe \
--model_type $model_type \
--task_name $task_name \
--data_basepath /pfm/data/bfm_benchmark \
--loadcheck_path $loadcheck_path \
--fp16 \
--ft \
--strategy Single \
--train_batch_size 32 \
--val_batch_size 32 \
--max_tokens 2048 \
--max_length 2048 \
--use_rd \
--rd_scale 1.0 \
--head_dropout 0.1 \
--log_interval 10 \
--down_stream_set $down_stream_set \
--spm_model_path /blob/shufxi/data/biofm/ur50bpe/ur50bpe.model
