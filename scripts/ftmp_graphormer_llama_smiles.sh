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
[ -z "${act_dropout}" ] && act_dropout=0.0
[ -z "${attn_dropout}" ] && attn_dropout=0.0
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=2e-5
[ -z "${total_num_steps}" ] && total_num_steps=10000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600

[ -z "${data_path}" ] && data_path='/home/peiran/FMproj/chemical-copilot-20230724'
# [ -z "${dataset_names}" ] && dataset_names='tdc'
# [ -z "${dataset_splits}" ] && dataset_splits='all-instruction'
[ -z "${dataset_names}" ] && dataset_names='mol-instruction-mol-desc'
[ -z "${dataset_splits}" ] && dataset_splits='clean'
[ -z "${dataset_ratios}" ] && dataset_ratios='1.0,1.0'
[ -z "${pool_mode}" ] && pool_mode='full'
[ -z "${embedding_length}" ] && embedding_length=20
[ -z "${model_max_length}" ] && model_max_length=512

[ -z "${loadcheck_path}" ] && loadcheck_path="."
[ -z "${save_dir}" ] && save_dir='/home/peiran/FMproj/output/llama2'
[ -z "${smiles_dict_path}" ] && smiles_dict_path="/home/peiran/FMproj/chemical-copilot/mol2idx_dict.jsonl"
[ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/home/peiran/FMproj/DiffTM100M/checkpoint7_new.pt"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/home/peiran/FMproj/MetaLLM-converted/7B-pp"
[ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/home/peiran/FMproj/llama2/llama-2-7b"
[ -z "${mol_size_path}" ] && mol_size_path="/home/peiran/FMproj/chemical-copilot/mol_size_dict.pkl"

[ -z "${add_3d}" ] && add_3d=false
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=4
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=1
[ -z "${zero_strategy}" ] && zero_strategy=1

[ -z "${micro_batch_size}" ] && micro_batch_size=2
[ -z "${global_batch_size}" ] && global_batch_size=64
[ -z "${max_position_embeddings}" ] && max_position_embeddings=2048
[ -z "${vocab_size}" ] && vocab_size=32000
[ -z "${vocabtokenizer_model_size}" ] && tokenizer_model="/home/peiran/FMproj/llama2/llama-2-7b/tokenizer.model"
[ -z "${llm_hidden_size}" ] && llm_hidden_size=4096


[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=6666
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
echo "micro_batch_size: ${micro_batch_size}"
echo "global_batch_size: ${global_batch_size}"
echo "max_position_embeddings: ${max_position_embeddings}"
echo "llm_hidden_size: ${llm_hidden_size}"


DISTRIBUTED_ARGS="
    --nproc_per_node $n_gpu \
    --master_port $MASTER_PORT
"
    # --nnodes $NNODES \
    # --node_rank $NODE_RANK \


DS_CONFIG=$save_dir/deepspeed.json

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
  }
}
EOT

ds_args=" --deepspeed --deepspeed_config=$DS_CONFIG"


wandb login --relogin 5d03b7a46d10f86ff45c4aedc570660a523edc0b

# torchrun $DISTRIBUTED_ARGS sfm/tasks/generalist/ft3d_graphormer_llama_inst.py \

deepspeed --num_gpu=4 --master_port=$MASTER_PORT sfm/tasks/generalist/ft3d_graphormer_llama_inst.py \
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
          --pipeline-model-parallel-size $pipeline_model_parallel_size \
          --tensor-model-parallel-size $tensor_model_parallel_size \
          --seed 666667 \
          --ft --fp16 \
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
          --embedding_length $embedding_length \
          --model_max_length $model_max_length \
          --micro-batch-size $micro_batch_size --global-batch-size $global_batch_size \
          --num-layers $layers --hidden-size $llm_hidden_size --seq-length $max_position_embeddings\
          --max-position-embeddings $max_position_embeddings --num-attention-heads $num_head \
          --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model $tokenizer_model \
          --no-query-key-layer-scaling  --attention-dropout 0 --hidden-dropout 0 \
          --use-rotary-position-embeddings --disable-bias-linear --seq-length 2048 \
          $ds_args


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
#     --deepspeed --deepspeed_config ./config_file/ds_config_ft.json \
#     --smiles_dict_path $smiles_dict_path \
#     --mol_size_path $mol_size_path \
#     --llm_model_name_or_path $llm_model_name_or_path \
#     --loadmfmcheck_path $loadmfmcheck_path \
#     --dataset_names $dataset_names \
#     --dataset_splits $dataset_splits
# fi


sleep inf
sleep inf
sleep inf
