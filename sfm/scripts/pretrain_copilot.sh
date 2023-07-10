#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=15
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
[ -z "${hidden_size}" ] && hidden_size=384
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${llm_hidden_size}" ] && llm_hidden_size=256
[ -z "${llm_ffn_size}" ] && llm_ffn_size=256
[ -z "${num_head}" ] && num_head=32
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mask_ratio}" ] && mask_ratio=0.3
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${total_num_steps}" ] && total_num_steps=100000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=6000

[ -z "${data_path}" ] && data_path='/home/peiran/FMproj/chemical-copilot/'
[ -z "${dataset_names}" ] && dataset_names='mt-data-1m'
[ -z "${dataset_splits}" ] && dataset_splits='train'
[ -z "${loadcheck_path}" ] && loadcheck_path='/home/peiran/FMproj/output/global_step1/'
[ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path='/home/peiran/FMproj/output'
[ -z "${loadllmcheck_path}" ] && loadllmcheck_path='/home/peiran/FMproj/output'
# [ -z "${output_path}" ] && output_path='/blob/ds_dataset/output/checkpoints'
[ -z "${output_path}" ] && output_path='/home/peiran/FMproj/output'
[ -z "${smiles_dict_path}" ] && smiles_dict_path="/home/peiran/FMproj/chemical-copilot/mol2idx_dict.jsonl"
[ -z "${model_name_or_path}" ] && model_name_or_path="/home/peiran/FMproj/MetaLLM-converted/7B"
[ -z "${mol_size_path}" ] && mol_size_path="/home/peiran/FMproj/chemical-copilot/mol_size_dict.pkl"

[ -z "${dataset_name}" ] && dataset_name="PM6-Full-3D"
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_parallelism}" ] && pipeline_parallelism=4

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=6669
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
echo "num_pred_attn_layer: ${num_pred_attn_layer}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "d_tilde: ${d_tilde}"
echo "sandwich_ln: ${sandwich_ln}"
echo "max_lr: ${max_lr}"
echo "total_num_steps: ${total_num_steps}"
echo "warmup_num_steps: ${warmup_num_steps}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "droppath_prob: ${droppath_prob}"
echo "atom_loss_coeff: ${atom_loss_coeff}"
echo "pos_loss_coeff: ${pos_loss_coeff}"
echo "no_2d: ${no_2d}"
echo "add_3d: ${add_3d}"

echo "data_path: ${data_path}"
echo "output_path: ${output_path}"
echo "smiles_dict_path: ${smiles_dict_path}"
echo "model_name_or_path: ${model_name_or_path}"

echo "dataset_name: ${dataset_name}"
echo "noise_scale: ${noise_scale}"
echo "mask_ratio: ${mask_ratio}"
echo "pipeline_parallelism: ${pipeline_parallelism}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1


# if [ $LOCAL_RANK == 0 ]; then
#   bash install.sh
# fi


# # # # # # # # # # # # # # # run PCQM4M-LSC-V2-3D
if [ $OMPI_COMM_WORLD_RANK == 0 ]; then
  sleep 600
  deepspeed --force_multi --hostfile=$hostfile train_copilot.py \
    --num-classes 128 \
    --encoder_attention_heads $num_head \
    --encoder_layers $layers \
    --encoder_ffn_embed_dim $ffn_size \
    --encoder_embed_dim $hidden_size \
    --llm_hidden_size $llm_hidden_size \
    --llm_ffn_size $llm_ffn_size \
    --droppath_prob $droppath_prob \
    --attn_dropout $attn_dropout \
    --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
    --sandwich_ln \
    --dataset-name $dataset_name \
    --data_path $data_path \
    --output_path $output_path \
    --dataset_names $dataset_names \
    --dataset_splits $dataset_splits \
    --pipeline_parallelism $pipeline_parallelism \
    --seed 666666 \
    --add-3d \
    --mask_ratio $mask_ratio \
    --noise_scale $noise_scale \
    --num_pred_attn_layer $num_pred_attn_layer \
    --d_tilde $d_tilde \
    --max_lr $max_lr \
    --total_num_steps $total_num_steps \
    --warmup_num_steps $warmup_num_steps \
    --deepspeed_config ./config_file/ds_config.json \
    --smiles_dict_path $smiles_dict_path \
    --mol_size_path $mol_size_path \
    --model_name_or_path $model_name_or_path \
    --loadcheck_path $loadcheck_path \
    --loadmfmcheck_path $loadmfmcheck_path \
    --loadllmcheck_path $loadllmcheck_path
fi

# sleep 100

# # # # # # single node
# # # # deepspeed --force_multi --num_node=$OMPI_COMM_WORLD_SIZE --num_gpus=$n_gpu --hostfile $hostfile train.py \
# # #   # deepspeed --force_multi --hostfile=$hostfile inference.py \
# # CUDA_VISIBLE_DEVICES=2,3


# # # # # # # # # # CUDA_VISIBLE_DEVICES=0,1,2,3
# deepspeed --num_gpus=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_copilot.py \
#     --num-classes 128 \
#     --encoder_attention_heads $num_head \
#     --encoder_layers $layers \
#     --encoder_ffn_embed_dim $ffn_size \
#     --encoder_embed_dim $hidden_size \
#     --llm_hidden_size $llm_hidden_size \
#     --llm_ffn_size $llm_ffn_size \
#     --droppath_prob $droppath_prob \
#     --attn_dropout $attn_dropout \
#     --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
#     --sandwich_ln \
#     --dataset-name $dataset_name \
#     --data_path $data_path \
#     --output_path $output_path \
#     --dataset_names $dataset_names \
#     --dataset_splits $dataset_splits \
#     --pipeline_parallelism $pipeline_parallelism \
#     --seed 666666 \
#     --add-3d \
#     --mask_ratio $mask_ratio \
#     --noise_scale $noise_scale \
#     --num_pred_attn_layer $num_pred_attn_layer \
#     --d_tilde $d_tilde \
#     --max_lr $max_lr \
#     --total_num_steps $total_num_steps \
#     --warmup_num_steps $warmup_num_steps \
#     --deepspeed_config ./config_file/eval_config.json \
#     --smiles_dict_path $smiles_dict_path \
#     --mol_size_path $mol_size_path \
#     --model_name_or_path $model_name_or_path \
#     --loadcheck_path $loadcheck_path \
#     --loadmfmcheck_path $loadmfmcheck_path \
#     --loadllmcheck_path $loadllmcheck_path

# torchrun --master_port=$MASTER_PORT $ddp_options --master_port=12345 train.py  \
#     --model_name_or_path "./${FINETUNE_FROM}" \
#     --data_path "./moleculenet-data/data/"  \
#     --dataset_names "${DATASET_NAMES}" \
#     --smiles_dict_path "./moleculenet-data/data/mol2idx_dict.jsonl" \
#     --model_molrep_dict_path "./moleculenet-data/processed/merged/" \
#     --output_dir "/blob/checkpoints/MetaLLM-${MODEL_SIZE}-D_NODE${D_NODE}-D_PROC${D_PROC}-${OUT_DIR_SUFFIX}" \
#     --num_train_epochs 3  \
#     --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
#     --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE}  \
#     --gradient_accumulation_steps ${GRADIENT_ACC}  \
#     --evaluation_strategy "no"  \
#     --save_strategy "steps"  \
#     --save_steps ${SAVE_STEPS}  \
#     --save_total_limit 1 \
#     --learning_rate ${LEARNING_RATE}  \
#     --weight_decay 0.  \
#     --warmup_ratio 0.03  \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 ${GRADIENT_CHECK} ${USE_LORA} ${USE_8BIT} \
#     --ddp_timeout 18000 \
#     --deepspeed "./configs/default_offload_opt_param_batch_size.json"

# # # run PCQM4M-LSC-V2-3D PP
# if (( $OMPI_COMM_WORLD_RANK == 0))
# then
#   deepspeed --num_node=$OMPI_COMM_WORLD_SIZE --num_gpus=$n_gpu --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py \
#             --num-classes 128 \
#             --encoder_attention_heads $num_head \
#             --encoder_layers $layers \
#             --encoder_ffn_embed_dim $ffn_size \
#             --encoder_embed_dim $hidden_size \
#             --add-3d $add_3d \
#             --no-2d $no_2d \
#             --dataset-name $dataset_name \
#             --data_path $data_path \
#             --output_path $output_path \
#             --pipeline_parallelism $pipeline_parallelism \
#             --deepspeed --deepspeed_config ./config_file/ds_config_stage1.json
# else
#   sleep inf
#   sleep inf
#   sleep inf
# fi

sleep inf
sleep inf
sleep inf
