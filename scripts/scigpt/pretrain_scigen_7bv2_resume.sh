#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${model_type}" ] && model_type="scigpt_7b"
[ -z "${weight_decay}" ] && weight_decay=0.1 # same as LLAMA2
[ -z "${max_lr}" ] && max_lr=1e-5  # LLAMA2 use 3e-4, let's use smaller lr
[ -z "${beta1}" ] && beta1=0.9 # same as LLAMA2
[ -z "${beta2}" ] && beta2=0.95 # same as LLAMA2
[ -z "${total_num_steps}" ] && total_num_steps=69760 # 6976 * 10 Epochs
[ -z "${warmup_num_steps}" ] && warmup_num_steps=1000 # same as LLAMA2
[ -z "${grad_scaler_init}" ] && grad_scaler_init=1
# LLAMA use 4M tokens per batch, we have context length 4096, so we use 4M/4k = 1k sents per batch
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024

# When use 128GPU and pipeline stage 16, the acc steps is 1k/(128/16) = 128
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=128
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=200
[ -z "${val_batch_interval}" ] && val_batch_interval=100000
[ -z "${log_interval}" ] && log_interval=10
[ -z "${epochs}" ] && epochs=10

[ -z "${strategy}" ] && strategy=Pipeline

[ -z "${vocab_size}" ] && vocab_size=38991
[ -z "${pad_token_id}" ] && pad_token_id=32000
[ -z "${max_position_embeddings}" ] && max_position_embeddings=4096
[ -z "${train_data_path}" ] && train_data_path='/hai1/shufxi/data/scigpt/v2/train.npy'
[ -z "${valid_data_path}" ] && valid_data_path='/hai1/shufxi/data/scigpt/v1/valid_sample.npy'
[ -z "${loadcheck_path}" ] && loadcheck_path='/hai1/shufxi/scigpt/7bv2/stageB/global_step26999/'
[ -z "${save_dir}" ] && save_dir='/hai1/shufxi/scigpt/7bv2/stageB_resume'
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=16
[ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="LlamaDecoderLayerPP"

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/amlarc-hosts/hostfile'
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

# echo "AZUREML_EXPERIMENT_ID: ${AZUREML_EXPERIMENT_ID}"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "model_type: ${model_type}"
echo "max_lr: ${max_lr}"
echo "total_num_steps: ${total_num_steps}"
echo "warmup_num_steps: ${warmup_num_steps}"
echo "weight_decay: ${weight_decay}"
echo "train_data_path: ${train_data_path}"
echo "valid_data_path: ${valid_data_path}"
echo "save_dir: ${save_dir}"
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1

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

set -x
mkdir -p "$save_dir"

torchrun $DISTRIBUTED_ARGS sfm/tasks/scigpt/pretrain_scigpt.py \
      --model_type "$model_type" \
      --vocab_size "$vocab_size" --pad_token_id "$pad_token_id" \
      --max_position_embeddings "$max_position_embeddings" \
      --train_data_path "$train_data_path" \
      --valid_data_path "$valid_data_path" \
      --weight_decay "$weight_decay" \
      --save_dir "$save_dir" \
      --seed 666668 \
      --bf16 \
      --grad_scaler_init "$grad_scaler_init" \
      --max_lr "$max_lr" \
      --beta1 "$beta1" --beta2 "$beta2" \
      --total_num_steps "$total_num_steps" \
      --warmup_num_steps "$warmup_num_steps" \
      --train_batch_size "$train_batch_size" \
      --val_batch_size "$val_batch_size" \
      --gradient_accumulation_steps "$gradient_accumulation_steps" \
      --save_epoch_interval "$save_epoch_interval" \
      --total_num_epochs "$epochs" \
      --save_batch_interval "$save_batch_interval" \
      --val_batch_interval "$val_batch_interval" \
      --log_interval "$log_interval" \
      --strategy "$strategy" \
      --pipeline_model_parallel_size "$pipeline_model_parallel_size" \
      --pp_partition_layer_name "$pp_partition_layer_name" \
      --load_ckpt --pretrained_ckpt_path "$loadcheck_path" \
      --ifresume


sleep inf
sleep inf
sleep inf
