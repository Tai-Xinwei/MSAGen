#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${beta1}" ] && beta1=0.9
[ -z "${beta2}" ] && beta2=0.98
[ -z "${weight_decay}" ] && weight_decay=0
[ -z "${max_lr}" ] && max_lr=0.0003
[ -z "${total_num_steps}" ] && total_num_steps=1000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=8000
[ -z "${train_batch_size}" ] && train_batch_size=2048
[ -z "${val_batch_size}" ] && val_batch_size=2048
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000000
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=300
[ -z "${strategy}" ] && strategy=DDP
[ -z "${train_data_path}" ] && train_data_path='/blob/shufxi/data/biofm/ur50bpe/train.npy'
[ -z "${valid_data_path}" ] && valid_data_path='/blob/shufxi/data/biofm/ur50bpe/valid.npy'
[ -z "${loadcheck_path}" ] && loadcheck_path="."
[ -z "${save_dir}" ] && save_dir='/mnt/output/'
[ -z "${mask_prob}" ] && mask_prob=0.15
[ -z "${initializer_range}" ] && initializer_range=0.02
[ -z "${grad_scaler_init}" ] && grad_scaler_init=256

[ -z "${wandb_group}" ] && wandb_group=scigpt
[ -z "${wandb_team}" ] && wandb_team=icuppjin
[ -z "${wandb_project}" ] && wandb_project=scigpt

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1



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

export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

wandb login --relogin "$WANDB_API_KEY"


if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
    DISTRIBUTED_ARGS=""
else
    if (( $OMPI_COMM_WORLD_SIZE == 1))
    then
        DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                    --master_port $MASTER_PORT"
    else
        DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                        --nnodes $OMPI_COMM_WORLD_SIZE \
                        --node_rank $OMPI_COMM_WORLD_RANK \
                        --master_addr $MASTER_ADDR"
    fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"


torchrun $DISTRIBUTED_ARGS sfm/tasks/pfm/pretrain_pfm_mlm_bpe.py \
        --model_type pfm_mlm_base \
        --train_data_path $train_data_path \
        --valid_data_path $valid_data_path \
        --weight_decay $weight_decay \
        --save_dir $save_dir \
        --seed 666666 \
        --fp16 \
        --beta1 $beta1 --beta2 $beta2 \
        --max_lr $max_lr \
        --mask_prob $mask_prob \
        --initializer_range $initializer_range \
        --total_num_steps $total_num_steps \
        --warmup_num_steps $warmup_num_steps \
        --train_batch_size $train_batch_size \
        --val_batch_size $val_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_epoch_interval $save_epoch_interval \
        --total_num_epochs $epochs \
        --save_batch_interval $save_batch_interval \
        --log_interval $log_interval \
        --strategy $strategy \
        --grad_scaler_init $grad_scaler_init \
        --use_rd --rd_scale 1.0 \
        --use_aa_loss --bpe2aa_path /blob/shufxi/data/biofm/ur50bpe/ur50bpe.bpe2aa.npz

sleep inf
sleep inf
sleep inf
