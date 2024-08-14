#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${weight_decay}" ] && weight_decay=0.1 # same as LLAMA2
[ -z "${max_lr}" ] && max_lr=1.5e-4  # LLAMA2 use 3e-4, let's use smaller lr
[ -z "${beta1}" ] && beta1=0.9 # same as LLAMA2
[ -z "${beta2}" ] && beta2=0.95 # same as LLAMA2
[ -z "${total_num_steps}" ] && total_num_steps=50000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=3000
[ -z "${grad_scaler_init}" ] && grad_scaler_init=1
[ -z "${unfreeze_param_list}" ] && unfreeze_param_list=""
[ -z "${learnable_cutoff}" ] && learnable_cutoff=0

# In this stage, the grad is too large to use grad accumulation
[ -z "${strategy}" ] && strategy=ThreeD
[ -z "${train_batch_size}" ] && train_batch_size=32
[ -z "${val_batch_size}" ] && val_batch_size=$train_batch_size
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=1
[ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="LlamaDecoderLayer"

[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=10

[ -z "${data_dir}" ] && data_dir=''
[ -z "${dict_path}" ] && dict_path='/data/peiran/blob/sfmdataeastus2/nlm/llama/Meta-Llama-3-8B/original/'
[ -z "${train_data_path}" ] && train_data_path='/data/peiran/blob/sfmdataeastus2/nlm/peiran/llama3_processed_data/lmdb/v5_valid_split/v5_protein_valid.npy.lmdb'
[ -z "${valid_data_path}" ] && valid_data_path='/data/peiran/blob/sfmdataeastus2/nlm/peiran/llama3_processed_data/lmdb/v5_valid_split/v5_protein_valid.npy.lmdb'
[ -z "${data_ratio}" ] && data_ratio=""
[ -z "${loadcheck_path}" ] && loadcheck_path='/data/peiran/blob/hai1data/sfm/llama/Meta-Llama-3-8B/original'
[ -z "${save_dir}" ] && save_dir='/data/peiran/output/'

train_data_files_list=(
  "DNA/archaea0_20000.lmdb"
  "DNA/archaea20000_-1.lmdb"
  "DNA/bacteria0_10000.lmdb"
  "DNA/bacteria10000_-1.lmdb"
  "DNA/fungi0_2000.lmdb"
  "DNA/fungi2000_-1.lmdb"
  "DNA/invertebrate0_60.lmdb"
  "DNA/invertebrate60_120.lmdb"
  "DNA/invertebrate120_180.lmdb"
  "DNA/invertebrate180_240.lmdb"
  "DNA/invertebrate240_300.lmdb"
  "DNA/invertebrate300_-1.lmdb"
  "DNA/mammals_binary_mammals0_30.lmdb"
  "DNA/mammals_binary_mammals30_60.lmdb"
  "DNA/mammals_binary_mammals60_90.lmdb"
  "DNA/mammals_binary_mammals90_120.lmdb"
  "DNA/mammals_binary_mammals120_150.lmdb"
  "DNA/mammals_binary_mammals150_180.lmdb"
  "DNA/mammals_binary_mammals180_-1.lmdb"
  "DNA/viruses0_-1.lmdb"
  "DNA/protozoa0_90.lmdb"
  "RNA/rnacentral.lmdb"
  "RNA/mRNAsub_75B.lmdb"
  "SFMAdditional.20240718.train.npy.lmdb"
  "SlimPajama_train_sample_300B_part00"
  "SlimPajama_train_sample_300B_part01"
  "SlimPajama_train_sample_300B_part02"
  "SlimPajama_train_sample_300B_part03"
  "SlimPajama_train_sample_300B_part04"
  "SlimPajama_train_sample_300B_part05"
  "SlimPajama_train_sample_300B_part06"
  "SlimPajama_train_sample_300B_part07"
  "SlimPajama_train_sample_300B_part08"
  "SlimPajama_train_sample_300B_part09"
  "SlimPajama_train_sample_300B_part10"
  "antibody.lmdb"
  "binary_train_pmc.npy.lmdb"
  "binary_train_pubmed.npy.lmdb"
  "binary_train_wrapped_pmc_0_200.npy.lmdb"
  "binary_train_wrapped_pmc_200_400.npy.lmdb"
  "binary_train_wrapped_pmc_400_600.npy.lmdb"
  "binary_train_wrapped_pmc_600_800.npy.lmdb"
  "binary_train_wrapped_pmc_800_1000.npy.lmdb"
  "binary_train_wrapped_pubmed_1_200.npy.lmdb"
  "binary_train_wrapped_pubmed_200_400.npy.lmdb"
  "binary_train_wrapped_pubmed_400_600.npy.lmdb"
  "binary_train_wrapped_pubmed_600_800.npy.lmdb"
  "binary_train_wrapped_pubmed_800_1115.npy.lmdb"
  "material.lmdb"
  "mgy_mgy_cluster.train.part1_3.lmdb"
  "mgy_mgy_cluster.train.part2_3.lmdb"
  "mgy_mgy_cluster.train.part3_3.lmdb"
  "text_and_material.lmdb"
  "train_c4.npy.lmdb"
  "train_patent.npy.lmdb"
  "train_scitext.npy.lmdb"
  "train_table.npy.lmdb"
  "train_text2prot.npy.lmdb"
  "ur90_2024_02.train.lmdb"
  "v6_processed_general_cmpd_text_train.npy.lmdb"
  "v6_processed_general_cmpd_train.npy.lmdb"
  "v6_processed_reagent_train.npy.lmdb"
  "v6_processed_zinc_3d_sub1_50B_train.npy.lmdb"
)
train_data_files=$(IFS=,; echo "${train_data_files_list[*]}")


valid_data_files_list=(
  "DNA/archaea0_20000valid.lmdb"
  "DNA/bacteria0_10000valid.lmdb"
  "DNA/fungi0_2000valid.lmdb"
  "DNA/invertebrate0_60valid.lmdb"
  "DNA/mammals_binary_mammals0_30valid.lmdb"
  "DNA/viruses0_-1valid.lmdb"
  "DNA/protozoa0_90valid.lmdb"
  "RNA/refseq_rnaValid.lmdb"
  "RNA/rnacentral_valid.lmdb"
  "antibody_full_seq_rmdup.sample30m.valid.pended.20240731.ab.txt.lmdb"
  "binary_valid_wrapped_pubmed_1_200.npy.lmdb"
  "mgy_clusters.pended.seq.valid.txt.lmdb"
  "valid.uniref90.shuf.10k.lmdb"
  "valid_c4.npy.lmdb"
  "valid_text_and_text2protein.npy.lmdb"
)
valid_data_files=$(IFS=,; echo "${valid_data_files_list[*]}")

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${wandb_group}" ] && wandb_group=nlm_llama3_stageA
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=nlm_llama3
[ -z "${wandb_key}" ] && wandb_key=local-094f941ede8eda7a00c307f50595f054be5382f7

export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
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


MEGATRON_ARGS="--micro-batch-size $micro_batch_size --global-batch-size $global_batch_size \
  --num-layers $layers --hidden-size $llm_hidden_size --seq-length $max_position_embeddings \
  --max-position-embeddings $max_position_embeddings --num-attention-heads $num_head \
  --seq-length $max_position_embeddings --disable-bias-linear --no-position-embedding --no-query-key-layer-scaling"

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
torchrun $DISTRIBUTED_ARGS sfm/tasks/nlm/pretrain_nlm3d.py \
      --model_type "$model_type" \
      --dict_path "$dict_path" \
      --vocab_size 38078 \
      --data_dir "$data_dir" \
      --train_data_path "$train_data_files" \
      --train_data_dir "$train_data_path" \
      --valid_data_path "$valid_data_files" \
      --valid_data_dir "$valid_data_path" \
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
      --pipeline_model_parallel_size "$pipeline_model_parallel_size" \
      --tensor_model_parallel_size "$tensor_model_parallel_size" \
      --pp_partition_layer_name "$pp_partition_layer_name" \
      --pretrained_ckpt_path "$loadcheck_path" \
      --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
      --unfreeze_param_list $unfreeze_param_list --learnable_cutoff $learnable_cutoff \
      ${MEGATRON_ARGS} ${load_ckpt} ${weighted_dataset}
