#!/bin/bash

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${model_type}" ] && model_type='scigpt_7b'
[ -z "${dict_path}" ] && dict_path='/hai1/ds_dataset/llama2/llama-2-7b'
[ -z "${max_position_embeddings}" ] && max_position_embeddings=4096
[ -z "${task_name}" ] && task_name='solubility'
[ -z "${data_basepath}" ] && data_basepath='/pfm/data/bfm_benchmark'
[ -z "${save_dir}" ] && save_dir='checkpoints/scigpt_7bv2_protein'
[ -z "${seed}" ] && seed=42
[ -z "${max_lr}" ] && max_lr=1e-5
[ -z "${beta1}" ] && beta1=0.9
[ -z "${beta2}" ] && beta2=0.95
[ -z "${train_batch_size}" ] && train_batch_size=1
[ -z "${val_batch_size}" ] && val_batch_size=1
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${total_num_epochs}" ] && total_num_epochs=10
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${pretrained_ckpt_path}" ] && pretrained_ckpt_path='/hai1/shufxi/scigpt/7bv2/stageB/global_step26999/'
[ -z "${max_length}" ] && max_length=2048

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

set -x

torchrun $DISTRIBUTED_ARGS \
    sfm/tasks/scigpt/finetune_scigpt_protein.py \
    --model_type "$model_type" \
    --dict_path "$dict_path" \
    --max_position_embeddings "$max_position_embeddings" \
    --task_name "$task_name" \
    --data_basepath "$data_basepath" \
    --save_dir "$save_dir" \
    --seed "$seed" \
    --fp16 \
    --max_lr "$max_lr" \
    --beta1 "$beta1" \
    --beta2 "$beta2" \
    --train_batch_size "$train_batch_size" \
    --val_batch_size "$val_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --total_num_epochs "$total_num_epochs" \
    --strategy 'Pipeline' \
    --pipeline_model_parallel_size "$pipeline_model_parallel_size" \
    --pp_partition_layer_name 'LlamaDecoderLayerPP' \
    --load_ckpt --pretrained_ckpt_path "$pretrained_ckpt_path" \
    --log_interval 10 \
    --max_length "$max_length" \
