#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=24
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
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
[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=2e-5
[ -z "${total_num_steps}" ] && total_num_steps=10000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600

# [ -z "${data_path}" ] && data_path='/mnt/shiyu/dataset/chemical-copilot'
# [ -z "${data_path}" ] && data_path='/home/v-peiqizhi/data/chemical-copilot-20230724'
# [ -z "${data_path}" ] && data_path='/home/v-peiqizhi/SFM_framework/tmp_data/ESOL'
# [ -z "${data_path}" ] && data_path='/sfm/ds_dataset/qizhi_numerical/freesolv'
# [ -z "${data_path}" ] && data_path='/sfm/ds_dataset/qizhi_numerical/bbbp'
# [ -z "${data_path}" ] && data_path='/sfm/ds_dataset/qizhi_numerical/bace'
# [ -z "${data_path}" ] && data_path='/sfm/ds_dataset/qizhi_numerical/lipo'
# [ -z "${data_path}" ] && data_path='/sfm/ds_dataset/qizhi_numerical/hiv'
[ -z "${data_path}" ] && data_path='/sfm/ds_dataset/qizhi_numerical/molnet_3_reg_up'
# [ -z "${dataset_names}" ] && dataset_names='tdc'
# [ -z "${dataset_splits}" ] && dataset_splits='all-instruction'
[ -z "${dataset_names}" ] && dataset_names='mol-instruction-mol-desc'
# [ -z "${dataset_splits}" ] && dataset_splits='clean'
[ -z "${dataset_splits}" ] && dataset_splits='train'
[ -z "${dataset_ratios}" ] && dataset_ratios='1.0,1.0'
[ -z "${pool_mode}" ] && pool_mode='full'
[ -z "${embedding_length}" ] && embedding_length=20
[ -z "${model_max_length}" ] && model_max_length=512

[ -z "${loadcheck_path}" ] && loadcheck_path="."
# [ -z "${save_dir}" ] && save_dir='/mnt/shiyu/models/converted/llama2'
# [ -z "${save_dir}" ] && save_dir='/home/v-peiqizhi/SFM_framework/output/ESOL_training_0816'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/freesolv_training_0824'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/bbbp_training_0824'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/bace_training_0824'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/lipo_training_0826'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/hiv_training_0825'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/bbbp_training_mlp_0828'
# [ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/freesolv_only_lmloss_0829'
[ -z "${save_dir}" ] && save_dir='/sfm/ds_dataset/output/qizhi_ft/molnet_3_reg_up_0829'

[ -z "${smiles_dict_path}" ] && smiles_dict_path="/home/peiran/FMproj/chemical-copilot/mol2idx_dict.jsonl"
# [ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/mnt/shiyu/models/graphormer_ckpts/checkpoint7_new.pt"
[ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/home/v-peiqizhi/models/graphormer_ckpts/checkpoint7_new.pt"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/home/peiran/FMproj/MetaLLM-converted/7B-pp"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/mnt/shiyu/models/converted/llama-2-7b"
[ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/home/v-peiqizhi/models/converted/llama-2-7b"
[ -z "${mol_size_path}" ] && mol_size_path="/home/peiran/FMproj/chemical-copilot/mol_size_dict.pkl"
[ -z "${save_batch_interval}"] && save_batch_interval=1000

[ -z "${add_3d}" ] && add_3d=false
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=1
[ -z "${strategy}" ] && strategy=Pipeline

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12345
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
echo "dataset_name: ${dataset_name}"
echo "noise_scale: ${noise_scale}"
echo "mask_ratio: ${mask_ratio}"
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"
echo "tensor_model_parallel_size: ${tensor_model_parallel_size}"
echo "embedding_length: ${embedding_length}"
echo "pool_mode: ${pool_mode}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
# export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
# export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1

# wandb login --relogin a88403970290781c26d2d5a6c07fe56df2116fc4
export WANDB_API_KEY=a88403970290781c26d2d5a6c07fe56df2116fc4
# export WANDB_PROJECT=Num_Dec
# export WANDB_RUN_NAME=molnet_3_reg_upsample

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

mkdir -p $save_dir

DS_CONFIG=$save_dir/deepspeed.json

[ -z "${zero_strategy}" ] && zero_strategy=1
[ -z "${micro_batch_size}" ] && micro_batch_size=8
[ -z "${global_batch_size}" ] && global_batch_size=8

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $global_batch_size,
  "train_micro_batch_size_per_gpu": $micro_batch_size,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": $zero_strategy
  },
  "fp16": {
    "enabled": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0000,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.0
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_type": "linear",
      "total_num_steps": $total_num_steps,
      "warmup_max_lr": $max_lr,
      "warmup_num_steps": $warmup_num_steps
    }
  },
  "wandb": {
    "enabled": true,
    "team": "yourteam",
    "group": "yourgroup",
    "project": "Num_Dec",
  }
}
EOT

ds_args="--deepspeed_config $DS_CONFIG"
set -x
torchrun $DISTRIBUTED_ARGS sfm/tasks/generalist/ft_graphormer_llama_inst.py \
          --num_classes 1 \
          --encoder_attention_heads $num_head \
          --encoder_layers $layers \
          --encoder_ffn_embed_dim $ffn_size \
          --encoder_embed_dim $hidden_size \
          --droppath_prob $droppath_prob \
          --attn_dropout $attn_dropout \
          --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
          --sandwich_ln \
          --data_path $data_path \
          --pipeline_model_parallel_size $pipeline_model_parallel_size \
          --tensor_model_parallel_size $tensor_model_parallel_size \
          --seed 666667 \
          --ft \
          --d_tilde $d_tilde \
          --num_pred_attn_layer $num_pred_attn_layer \
          --max_lr $max_lr \
          --save_dir $save_dir \
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --loadcheck_path $loadcheck_path \
          --llm_model_name_or_path $llm_model_name_or_path \
          --loadmfmcheck_path $loadmfmcheck_path \
          --dataset_names $dataset_names \
          --dataset_splits $dataset_splits \
          --dataset_ratios $dataset_ratios \
          --pool_mode $pool_mode \
          --strategy $strategy \
          --embedding_length $embedding_length \
          --model_max_length $model_max_length \
          --pp_partition_layer_name "LlamaDecoderLayerPP" \
          --load_ckpt \
          --unfreeze_param_list "adaptor" \
          --save_batch_interval $save_batch_interval \
          --log_interval 100000000 \
          $ds_args | tee -a $save_dir/train.log


# if [ $OMPI_COMM_WORLD_RANK == 0 ]; then
#   sleep 600
#   deepspeed --force_multi --hostfile=$hostfile sfm/tasks/ft_graphormerllama.py \
#     --num-classes 1 \
#     --encoder_attention_heads $num_head \
#     --encoder_layers $layers \
#     --encoder_ffn_embed_dim $ffn_size \
#     --encoder_embed_dim $hidden_size \
#     --droppath_prob $droppath_prob \
#     --attn_dropout $attn_dropout \
#     --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
#     --sandwich_ln \
#     --dataset-name $dataset_name \
#     --data_path $data_path \
#     --output_path $output_path \
#     --pipeline_parallelism $pipeline_parallelism \
#     --seed 666667 \
#     --ft \
#     --d_tilde $d_tilde \
#     --num_pred_attn_layer $num_pred_attn_layer \
#     --max_lr $max_lr \
#     --output_path $output_path \
#     --total_num_steps $total_num_steps \
#     --warmup_num_steps $warmup_num_steps \
#     --loadcheck_path $loadcheck_path \
#     --deepspeed --deepspeed_config ./config_file/ds_config_pp.json \
#     --smiles_dict_path $smiles_dict_path \
#     --mol_size_path $mol_size_path \
#     --llm_model_name_or_path $llm_model_name_or_path \
#     --loadmfmcheck_path $loadmfmcheck_path \
#     --dataset_names $dataset_names \
#     --dataset_splits $dataset_splits \
#     --pool_mode $pool_mode \
#     --strategy $strategy \
#     --embedding_length $embedding_length \
#     --model_max_length $model_max_length \
#     --pp_partition_layer_name "LlamaDecoderLayerPP" \
#     --load_ckpt \
#     --unfreeze_param_list "adaptor,graphormer" \
#     --save_batch_interval $save_batch_interval
# fi

sleep inf
sleep inf
sleep inf
