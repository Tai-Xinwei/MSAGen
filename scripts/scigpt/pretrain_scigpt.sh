#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${vocab_size}" ] && vocab_size=33494

[ -z "${layers}" ] && layers=6
[ -z "${hidden_size}" ] && hidden_size=256
[ -z "${ffn_size}" ] && ffn_size=1024
[ -z "${num_head}" ] && num_head=16
[ -z "${num_key_value_heads}" ] && num_key_value_heads=16
[ -z "${max_length}" ] && max_length=2048

[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${max_lr}" ] && max_lr=4e-4
[ -z "${total_num_steps}" ] && total_num_steps=1000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600
[ -z "${train_batch_size}" ] && train_batch_size=4
[ -z "${val_batch_size}" ] && val_batch_size=4
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000000
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=1000

[ -z "${strategy}" ] && strategy=Pipeline

[ -z "${dict_path}" ] && dict_path='/mnt/protein/scigpt/sample/scigpt_llama2_unidict.txt'
[ -z "${train_data_path}" ] && train_data_path='/mnt/protein/scigpt/sample/scigpt.sample.train.txt'
[ -z "${valid_data_path}" ] && valid_data_path='/mnt/protein/scigpt/sample/scigpt.sample.valid.txt'
[ -z "${loadcheck_path}" ] && loadcheck_path="."
[ -z "${save_dir}" ] && save_dir='/mnt/output/'
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=2

[ -z "${wandb_group}" ] && wandb_group=scigpt
[ -z "${wandb_team}" ] && wandb_team=icuppjin
[ -z "${wandb_project}" ] && wandb_project=scigpt

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
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "max_lr: ${max_lr}"
echo "total_num_steps: ${total_num_steps}"
echo "warmup_num_steps: ${warmup_num_steps}"
echo "weight_decay: ${weight_decay}"
echo "train_data_path: ${train_data_path}"
echo "valid_data_path: ${valid_data_path}"
echo "output_path: ${output_path}"
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1

wandb login --relogin 5d03b7a46d10f86ff45c4aedc570660a523edc0b


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


torchrun $DISTRIBUTED_ARGS sfm/tasks/scigpt/pretrain_scigpt.py \
          --vocab_size $vocab_size \
          --dict_path $dict_path \
          --train_data_path $train_data_path \
          --valid_data_path $valid_data_path \
          --num_attention_heads $num_head \
          --num_key_value_heads $num_key_value_heads \
          --num_hidden_layers $layers \
          --hidden_size $hidden_size \
          --intermediate_size $ffn_size \
          --weight_decay $weight_decay \
          --save_dir $save_dir \
          --seed 666666 \
          --fp16 \
          --max_lr $max_lr \
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --max_position_embeddings $max_length \
          --train_batch_size $train_batch_size \
          --val_batch_size $val_batch_size \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --save_epoch_interval $save_epoch_interval \
          --total_num_epochs $epochs \
          --save_batch_interval $save_batch_interval \
          --log_interval $log_interval \
          --strategy $strategy \
          --pipeline_model_parallel_size $pipeline_model_parallel_size
          # --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project


sleep inf
sleep inf
sleep inf
