#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# In this stage, we only finetune new emb
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# For local test
# export WANDB_API_KEY=local-84c43c09161e2c012c3317ccb9becc6148001b8e
# export WANDB_PROJECT=nlm_phi35_guoqing
# export WANDB_TEAM=ai4s-sfm
# export WANDB_RUN_NAME=finetune_sfm_nlmphi35_inst_SFMMolInstruct.20240807_v2_dialogue_1vs1_bs2048_test
# export wandb_group=instruct
# export train_batch_size=2048
# export val_batch_size=2048
# export gradient_accumulation_steps=512 # 32
# export max_lr=2e-5
# export total_num_steps=20000
# export warmup_num_steps=300
# export epochs=12
# export train_hf_data_path=/home/guoqingliu/blob_nlm/guoqing/SFM_inst_tune_dataset/dialogue/train.merged.filt.out.tsv.filt.phi35mini.lmdb
# export hf_sample_count=-1
# export train_data_path=/home/guoqingliu/blob_nlm/guoqing/SFM_inst_tune_dataset/science/train.all.v2.tsv.phi35mini.lmdb
# export valid_data_path=/home/guoqingliu/blob_nlm/guoqing/SFM_inst_tune_dataset/science/overall.val.tsv.phi35mini.lmdb
# # export dict_path=/home/guoqingliu/blob_nlm/llama/llama-2-7b
# export dict_path=/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct
# export loadcheck_path=/home/guoqingliu/blob_nlm/phi/Phi-3.5-mini-instruct/pt/phi35mini_instruct.pt
# export save_dir=/home/guoqingliu/blob_nlm/output/phi35mini/SFMMolInstruct.20241028_v2_dialogue_1vs1_test/


[ -z "${weight_decay}" ] && weight_decay=0.1 # same as LLAMA2
[ -z "${max_lr}" ] && max_lr=2e-4  # LLAMA2 use 3e-4, let's use smaller lr
[ -z "${beta1}" ] && beta1=0.9 # same as LLAMA2
[ -z "${beta2}" ] && beta2=0.95 # same as LLAMA2
[ -z "${total_num_steps}" ] && total_num_steps=300000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=8000
[ -z "${grad_scaler_init}" ] && grad_scaler_init=1
# [ -z "${unfreeze_param_list}" ] && unfreeze_param_list="lm_head.weight,word_embeddings.weight"
[ -z "${unfreeze_param_list}" ] && unfreeze_param_list=""
[ -z "${learnable_cutoff}" ] && learnable_cutoff=0

# In this stage, the grad is too large to use grad accumulation
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${train_batch_size}" ] && train_batch_size=4
[ -z "${val_batch_size}" ] && val_batch_size=$train_batch_size
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=1
[ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="LlamaDecoderLayerMP"

[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=1000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=15
[ -z "${train_hf_data_path}" ] && train_hf_data_path=''
[ -z "${hf_sample_count}" ] && hf_sample_count=-1

[ -z "${dict_path}" ] && dict_path='/home/v-zekunguo/nlm/llama/Meta-Llama-3-8B/original'
[ -z "${train_data_path}" ] && train_data_path='/home/v-zekunguo/nlm/zekun/data/scidata/chembl/lmdb/t2d.test.csv.lmdb'
[ -z "${valid_data_path}" ] && valid_data_path='/home/v-zekunguo/nlm/zekun/data/scidata/chembl/lmdb/t2d.test.csv.lmdb'
[ -z "${loadcheck_path}" ] && loadcheck_path='/home/v-zekunguo/nlm/peiran/output/finetune_base_150B_G64/global_step28464/'
[ -z "${save_dir}" ] && save_dir='/home/v-zekunguo//nlm/zekun/output/1b/chembl_t2d_G256_bs256_lr2e5'

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${wandb_group}" ] && wandb_group=other
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=nlm_llama3_zekun
[ -z "${wandb_key}" ] && wandb_key=local-84c43c09161e2c012c3317ccb9becc6148001b8e

export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

# [ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${n_gpu}" ] && n_gpu=$(rocm-smi | grep -c '^[0-9]')

# OMPI_COMM_WORLD_SIZE=1 # for local test

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

export OMP_NUM_THREADS=16

torchrun $DISTRIBUTED_ARGS sfm/tasks/nlm/fintune_nlm_phi35mini_inst_amd.py \
      --model_type "$model_type" \
      --dict_path "$dict_path" \
      --train_data_path "$train_data_path" \
      --valid_data_path "$valid_data_path" \
      --weight_decay "$weight_decay" \
      --save_dir "$save_dir" \
      --seed 666666 \
      --learnable_cutoff "$learnable_cutoff" \
      --unfreeze_param_list "$unfreeze_param_list" \
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
      --train_hf_data_path "$train_hf_data_path" \
      --hf_sample_count "$hf_sample_count" \
      --pipeline_model_parallel_size "$pipeline_model_parallel_size" \
      --tensor_model_parallel_size "$tensor_model_parallel_size" \
      --load_ckpt --pretrained_ckpt_path $loadcheck_path \
      --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
      --ifresume
