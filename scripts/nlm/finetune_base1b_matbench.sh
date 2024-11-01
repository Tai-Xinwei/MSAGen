#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${weight_decay}" ] && weight_decay=0.1 # same as LLAMA2
[ -z "${max_lr}" ] && max_lr=2e-5  # LLAMA2 use 3e-4, let's use smaller lr
[ -z "${beta1}" ] && beta1=0.9 # same as LLAMA2
[ -z "${beta2}" ] && beta2=0.95 # same as LLAMA2
[ -z "${total_num_steps}" ] && total_num_steps=900000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=4000
[ -z "${grad_scaler_init}" ] && grad_scaler_init=1
# [ -z "${unfreeze_param_list}" ] && unfreeze_param_list="lm_head.weight,word_embeddings.weight"
# [ -z "${learnable_cutoff}" ] && learnable_cutoff=128256

# In this stage, the grad is too large to use grad accumulation
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${train_batch_size}" ] && train_batch_size=32
[ -z "${val_batch_size}" ] && val_batch_size=$train_batch_size
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=1

[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=4000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=10

[ -z "${data_dir}" ] && data_dir=''
[ -z "${dict_path}" ] && dict_path='/sfmdataeastus2/nlm/llama/Meta-Llama-3-8B/original/'
[ -z "${train_data_path}" ] && train_data_path='/msralaphilly2/ml-la/v-yantingli/matbench_1011/matbench_sfm/fold_4_train.tsv'
[ -z "${valid_data_path}" ] && valid_data_path='/msralaphilly2/ml-la/v-yantingli/matbench_1011/matbench_sfm/fold_4_valid.tsv'
[ -z "${data_ratio}" ] && data_ratio=""
[ -z "${loadcheck_path}" ] && loadcheck_path='/sfmdataeastus2/nlm/peiran/output/finetune_base_150B_G64/global_step28464/'
[ -z "${save_dir}" ] && save_dir='/msralaphilly2/ml-la/v-yantingli/nlm/matbench/sfm_1b_e20/fold_4'

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${wandb_group}" ] && wandb_group=nlm_llama3_matbench_sfm_1b
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=nlm_llama3_matbench
[ -z "${wandb_key}" ] && wandb_key=local-a5b2480774245970f207d971f17314f5a30c0021

export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

if [[ -z "${n_gpu}" ]]
then
    if command -v nvidia-smi &> /dev/null
    then
        n_gpu=$(nvidia-smi -L | wc -l)
    else
        n_gpu=$(rocm-smi | grep -c '^[0-9]')
    fi
fi

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
  world_size=1
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --master_port $MASTER_PORT"
    world_size=$n_gpu
  else
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
    world_size=$OMPI_COMM_WORLD_SIZE*$n_gpu
  fi
fi

# if [[ "${strategy}" == "ThreeD" ]]; then
dp_worldsize=$(($world_size/$pipeline_model_parallel_size/$tensor_model_parallel_size))
[ -z "${micro_batch_size}" ] && micro_batch_size=$(($train_batch_size/$gradient_accumulation_steps/$dp_worldsize))
[ -z "${num_head}" ] && num_head=32
[ -z "${global_batch_size}" ] && global_batch_size=$train_batch_size
[ -z "${max_position_embeddings}" ] && max_position_embeddings=8192
[ -z "${llm_hidden_size}" ] && llm_hidden_size=4096
[ -z "${layers}" ] && layers=24
[ -z "${num_head}" ] && num_head=32

# if load ckpt, default is False
[ -z "${load_ckpt}" ] && load_ckpt=False
if [[ "${load_ckpt}" == "True" ]]; then
  load_ckpt="--load_ckpt"
else
  load_ckpt=""
fi

[ -z "${weighted_dataset}" ] && weighted_dataset=False
if [[ "${weighted_dataset}" == "True" ]]; then
  weighted_dataset="--weighted_dataset"
else
  weighted_dataset=""
fi

echo -e "\n\n"
echo "==================================MP==========================================="
echo "n_gpu: ${n_gpu}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "LOCAL_RANK : ${LOCAL_RANK}"
echo "world_size : ${world_size}"
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
echo "data_ratio: ${data_ratio}"
echo "train_batch_size: ${train_batch_size}"
echo "val_batch_size: ${val_batch_size}"
echo "micro_batch_size: ${micro_batch_size}"
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"
echo "save_dir: ${save_dir}"
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"
echo "tensor_model_parallel_size: ${tensor_model_parallel_size}"

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"

wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key
export WANDB_API_KEY=$wandb_key

set -x
torchrun $DISTRIBUTED_ARGS sfm/tasks/nlm/finetune_nlm_1Bbase_matbench.py \
      --model_type "$model_type" \
      --dict_path "$dict_path" \
      --data_dir "$data_dir" \
      --train_data_path "$train_data_path" \
      --valid_data_path "$valid_data_path" \
      --train_data_ratio "$data_ratio" \
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
      --pretrained_ckpt_path "$loadcheck_path" \
      --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
      --ifresume \
      ${load_ckpt} ${weighted_dataset}
