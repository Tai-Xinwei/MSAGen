#!/bin/bash

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${train_data_path}" ] && train_data_path="/hai1/kaiyuan/scigpt/molinst/data/scigpt_molinst_pro_v3/train.tsv"
[ -z "${valid_data_path}" ] && valid_data_path='/hai1/kaiyuan/scigpt/molinst/data/scigpt_molinst_pro_v3/test.tsv'

[ -z "${total_num_epochs}" ] && total_num_epochs=1
[ -z "${train_batch_size}" ] && train_batch_size=256
[ -z "${val_batch_size}" ] && val_batch_size=32

num_lines=$(wc -l < "$train_data_path")
[ -z "${total_num_steps}" ] && total_num_steps=$((num_lines / train_batch_size * total_num_epochs))

[ -z "${model_type}" ] && model_type='scigpt_7b'
[ -z "${dict_path}" ] && dict_path='/hai1/ds_dataset/llama2/llama-2-7b'
[ -z "${max_position_embeddings}" ] && max_position_embeddings=4096
[ -z "${save_dir}" ] && save_dir="/blob/v-kehanwu/nlm/checkpoints/llama_inst/$(date +%Y%m%d%H%M%S)"
[ -z "${seed}" ] && seed=46
[ -z "${max_lr}" ] && max_lr=2e-5
[ -z "${beta1}" ] && beta1=0.9
[ -z "${beta2}" ] && beta2=0.999
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=32
[ -z "${warmup_num_steps}" ] && warmup_num_steps=100
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${pretrained_ckpt_path}" ] && pretrained_ckpt_path='/hai1/ds_dataset/llama2/llama-2-7b'
[ -z "${max_length}" ] && max_length=1024

[ -z "${wandb_group}" ] && wandb_group=NLM
[ -z "${wandb_team}" ] && wandb_team=HankerWu
[ -z "${wandb_project}" ] && wandb_project=sfm
[ -z "${wandb_key}" ] && wandb_key=140f5ace0c8e16afe6efe3921fa0d90d1c7a3e61

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
    sfm/tasks/llama2_inst/llama_instrcution_tuning.py \
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
    --pp_partition_layer_name 'LlamaDecoderLayerPP' \
    --load_ckpt --pretrained_ckpt_path "$pretrained_ckpt_path" \
    --log_interval 1 \
    --conditional_generation --use_template \
    --bf16 \
    --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
    --ifresume

echo "Done"
