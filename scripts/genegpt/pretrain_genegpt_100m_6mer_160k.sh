#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# In this stage, we only finetune new emb
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${model_type}" ] && model_type="genegpt_100m"
[ -z "${weight_decay}" ] && weight_decay=0.1 # same as LLAMA2
# TODO: we need grad clip
[ -z "${max_lr}" ] && max_lr=2e-4  # LLAMA2 use 3e-4, let's use smaller lr
[ -z "${beta1}" ] && beta1=0.9 # same as LLAMA2
[ -z "${beta2}" ] && beta2=0.95 # same as LLAMA2
[ -z "${total_num_steps}" ] && total_num_steps=30000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=2000
[ -z "${grad_scaler_init}" ] && grad_scaler_init=1
[ -z "${train_batch_size}" ] && train_batch_size=4
[ -z "${val_batch_size}" ] && val_batch_size=4
# [ -z "${unfreeze_param_list}" ] && unfreeze_param_list=""
[ -z "${learnable_cutoff}" ] && learnable_cutoff=0

# In this stage, the grad is too large to use grad accumulation
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=1000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=1

# [ -z "${strategy}" ] && strategy=ThreeD
[ -z "${strategy}" ] && strategy=Pipeline

[ -z "${dict_path}" ] && dict_path='/hai1/ds_dataset/llama2/llama-2-7b'
[ -z "${train_data_path}" ] && train_data_path='/home/v-zekunguo/data/v-zekunguo/gene/data/lmdb'
[ -z "${valid_data_path}" ] && valid_data_path='/home/v-zekunguo/data/v-zekunguo/gene/data/valid_lmdb'
# [ -z "${loadcheck_path}" ] && loadcheck_path='/hai1/ds_dataset/llama2/llama-2-7b'
[ -z "${save_dir}" ] && save_dir='/home/v-zekunguo/data/v-zekunguo/gene/checkpoints/1b6kmer16k'
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="LlamaDecoderLayerPP"


[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1



dp_worldsize=$(($world_size/$pipeline_model_parallel_size/$tensor_model_parallel_size))
[ -z "${micro_batch_size}" ] && micro_batch_size=$(($train_batch_size/$gradient_accumulation_steps/$dp_worldsize))
[ -z "${num_head}" ] && num_head=32
[ -z "${global_batch_size}" ] && global_batch_size=$train_batch_size
[ -z "${max_position_embeddings}" ] && max_position_embeddings=163840
[ -z "${tokens_per_sample}" ] && tokens_per_sample=163840
[ -z "${max_tokens}" ] && max_tokens=163480
[ -z "${llm_hidden_size}" ] && llm_hidden_size=4096
[ -z "${layers}" ] && layers=24
[ -z "${num_head}" ] && num_head=32

[ -z "${wandb_group}" ] && wandb_group=SFM
[ -z "${wandb_team}" ] && wandb_team=large-scale-pde
[ -z "${wandb_project}" ] && wandb_project=gene
[ -z "${wandb_key}" ] && wandb_key=local-84c43c09161e2c012c3317ccb9becc6148001b8e

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

wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key
export WANDB_API_KEY=$wandb_key


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
torchrun $DISTRIBUTED_ARGS sfm/tasks/genegpt/pretrain_genegpt.py \
      --model_type "$model_type" \
      --dict_path "$dict_path" \
      --train_data_path "$train_data_path" \
      --valid_data_path "$valid_data_path" \
      --weight_decay "$weight_decay" \
      --save_dir "$save_dir" \
      --seed 666666 \
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
      --log_interval "$log_interval" \
      --strategy "$strategy" \
      --max_tokens $max_tokens \
      --tokens_per_sample $tokens_per_sample \
      --pipeline_model_parallel_size "$pipeline_model_parallel_size" \
      --pp_partition_layer_name "$pp_partition_layer_name" \
      --load_ckpt --pretrained_ckpt_path "$loadcheck_path" \
      --unfreeze_param_list "$unfreeze_param_list" \
      --learnable_cutoff "$learnable_cutoff" \
      --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
      ${MEGATRON_ARGS}



sleep inf
sleep inf
sleep inf
