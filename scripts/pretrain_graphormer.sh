#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=15
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=3072
[ -z "${num_head}" ] && num_head=32
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${total_num_steps}" ] && total_num_steps=10000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=16
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=8
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1000
[ -z "${save_batch_interval}" ] && save_batch_interval=10000
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=1000

[ -z "${data_path}" ] && data_path='/home/peiran/FMproj/pm6-86m-3d-filter'
# [ -z "${data_path}" ] && data_path="/data/pm6-86m-3d-filter/pm6-86m-3d-filter"
[ -z "${loadcheck_path}" ] && loadcheck_path="."
[ -z "${save_dir}" ] && save_dir='/home/peiran/FMproj/output/'
# [ -z "${dataset_name}" ] && dataset_name="PCQM4M-LSC-V2-3D"
[ -z "${dataset_name}" ] && dataset_name="PM6-Full-3D"
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_parallelism}" ] && pipeline_parallelism=0
[ -z "${strategy}" ] && strategy=Zero2

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
# [ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=-1


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
echo "pipeline_parallelism: ${pipeline_parallelism}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1


# # # # # # run PCQM4M-LSC-V2-3D
# if [ $OMPI_COMM_WORLD_RANK == 0 ]; then
#   sleep 1200
#   deepspeed --force_multi --hostfile=$hostfile train.py \
#     --num-classes 128 \
#     --encoder_attention_heads $num_head \
#     --encoder_layers $layers \
#     --encoder_ffn_embed_dim $ffn_size \
#     --encoder_embed_dim $hidden_size \
#     --droppath_prob $droppath_prob \
#     --attn_dropout $attn_dropout \
#     --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
#     --sandwich_ln \
#     --dataset-name $dataset_name \
#     --data_path $data_path \
#     --output_path $output_path \
#     --pipeline_parallelism $pipeline_parallelism \
#     --seed 666667 \
#     --add-3d \
#     --mask_ratio $mask_ratio \
#     --noise_scale $noise_scale \
#     --num_pred_attn_layer $num_pred_attn_layer \
#     --d_tilde $d_tilde \
#     --max_lr $max_lr \
#     --total_num_steps $total_num_steps \
#     --warmup_num_steps $warmup_num_steps \
#     --loadcheck_path $loadcheck_path \
#     --deepspeed --deepspeed_config ./config_file/ds_config.json
# fi


# # # # # # # single node
# # # # # deepspeed --force_multi --num_node=$OMPI_COMM_WORLD_SIZE --num_gpus=$n_gpu --hostfile $hostfile train.py \
# deepspeed --num_gpus=$n_gpu train.py \
deepspeed --num_gpus=4 sfm/tasks/graphormer/pretrain_graphormer.py \
    --num_classes 128 \
    --encoder_attention_heads $num_head \
    --encoder_layers $layers \
    --encoder_ffn_embed_dim $ffn_size \
    --encoder_embed_dim $hidden_size \
    --droppath_prob $droppath_prob \
    --attn_dropout $attn_dropout \
    --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
    --sandwich_ln \
    --dataset_names $dataset_name \
    --data_path $data_path \
    --save_dir $save_dir \
    --pipeline_parallelism $pipeline_parallelism \
    --seed 666666 \
    --add_3d --fp16 \
    --mask_ratio $mask_ratio \
    --noise_scale $noise_scale \
    --num_pred_attn_layer $num_pred_attn_layer \
    --d_tilde $d_tilde \
    --max_lr $max_lr \
    --strategy $strategy \
    --total_num_steps $total_num_steps \
    --warmup_num_steps $warmup_num_steps \
    --train_batch_size $train_batch_size --val_batch_size $val_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_epoch_interval $save_epoch_interval --epochs $epochs \
    --save_batch_interval $save_batch_interval --log_interval $log_interval
    # --deepspeed_config ./config_file/ds_config.json


sleep inf
sleep inf
sleep inf
