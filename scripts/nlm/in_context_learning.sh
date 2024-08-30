#!/bin/bash

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${train_data_path}" ] && train_data_path=''
[ -z "${valid_data_path}" ] && valid_data_path=''
[ -z "${seed}" ] && seed=50
filename=$(basename "${train_data_path}")
base=${filename%%.*}

[ -z "${save_dir}" ] && save_dir="/mnt/msralaphilly2/v-yinzhezhou/model_output/${base}_seed=${seed}/$(date +%Y%m%d%H%M%S)"

[ -z "${total_num_epochs}" ] && total_num_epochs=1
[ -z "${train_batch_size}" ] && train_batch_size=128
[ -z "${val_batch_size}" ] && val_batch_size=128

num_lines=$(wc -l < "$train_data_path")
[ -z "${total_num_steps}" ] && total_num_steps=$((num_lines * total_num_epochs / train_batch_size ))

[ -z "${model_type}" ] && model_type='sfm_nlm_moe_8x7b'
[ -z "${dict_path}" ] && dict_path='/mnt/sfmdataeastus2/Mixtral-8x7B-v0.1'
[ -z "${max_position_embeddings}" ] && max_position_embeddings=8192


[ -z "${max_lr}" ] && max_lr=2e-5
[ -z "${beta1}" ] && beta1=0.9
[ -z "${beta2}" ] && beta2=0.999
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=128
[ -z "${warmup_num_steps}" ] && warmup_num_steps=10
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=8
[ -z "${pretrained_ckpt_path}" ] && pretrained_ckpt_path='/mnt/sfmdataeastus2/shufxi/nlm/8x7b/stageB_pp8_acc16_total1536_12m_bsz/global_step32000/'
[ -z "${max_length}" ] && max_length=1024
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
    sfm/tasks/nlm/finetune_nlm_moe_instruction.py \
    --model_type "$model_type" \
    --dict_path "$dict_path" \
    --train_data_path "$train_data_path" \
    --valid_data_path "$valid_data_path" \
    --max_position_embeddings "$max_position_embeddings" \
    --max_length "$max_length" \
    --save_dir "$save_dir" \
    --seed "$seed" \
    --max_lr "$max_lr" \
    --beta1 "$beta1" \
    --beta2 "$beta2" \
    --weight_decay 0 \
    --train_batch_size "$train_batch_size" \
    --val_batch_size "$val_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --total_num_epochs "$total_num_epochs" \
    --total_num_steps $total_num_steps \
    --warmup_num_steps $warmup_num_steps \
    --strategy 'Pipeline' \
    --pipeline_model_parallel_size "$pipeline_model_parallel_size" \
    --pp_partition_layer_name 'MoeDecoderLayerPP' \
    --load_ckpt --pretrained_ckpt_path "$pretrained_ckpt_path" \
    --log_interval 1 \
    --moe_impl "vanilla" \
    --conditional_generation --use_template \
    --bf16 \
    --ifresume

echo "Done"
