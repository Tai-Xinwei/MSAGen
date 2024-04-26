#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

[ -z "${layers}" ] && layers=33
[ -z "${hidden_size}" ] && hidden_size=1280
[ -z "${ffn_size}" ] && ffn_size=5120
[ -z "${num_head}" ] && num_head=20

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
[ -z "${total_num_steps}" ] && total_num_steps=100000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600
# training parameters for generalist
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=32
[ -z "${train_batch_size}" ] && train_batch_size=256
[ -z "${val_batch_size}" ] && val_batch_size=256
# [ -z "${train_batch_size}" ] && train_batch_size=32

[ -z "${train_data_path}" ] && train_data_path='/blob/v-kehanwu/data/progpt_data/progpt_instructions_train_bpe.lmdb'
[ -z "${valid_data_path}" ] && valid_data_path='/blob/v-kehanwu/data/progpt_data/progpt_instructions_test_bpe.lmdb'
[ -z "${pool_mode}" ] && pool_mode='full'
[ -z "${embedding_length}" ] && embedding_length=20
[ -z "${model_max_length}" ] && model_max_length=2048

[ -z "${loadcheck_path}" ] && loadcheck_path="."
[ -z "${save_dir}" ] && save_dir='/blob/v-kehanwu/nlm/checkpoints/'
[ -z "${loadbfmckpt_path}" ] && loadbfmckpt_path='/hai1.sfm/nlm/output/checkpoint_E144_new.pt'
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/hai1/ds_dataset/llama2/llama-2-7b"
[ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/blob/v-kehanwu/SFM/scigpt/stageB.prot/global_step224655"
[ -z "${finetune_from_checkpoint_dir}" ] && finetune_from_checkpoint_dir="/blob/v-kehanwu/nlm/checkpoints/bfm_scigpt_prot"
[ -z "${finetune_from_checkpoint_id}" ] && finetune_from_checkpoint_id="global_step20499"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/blob/v-kehanwu/nlm/checkpoints/bfm_scigpt_prot/global_step20499"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/blob/v-kehanwu/SFM/scigpt/stageB/global_step32999"
[ -z "${tokenizer_path}" ] && tokenizer_path="/blob/shufxi/data/scigpt"

[ -z "${save_batch_interval}"] && save_batch_interval=500
[ -z "${log_interval}" ] && log_interval=20

[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=1
[ -z "${strategy}" ] && strategy=Pipeline
[ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="manual"
# [ -z "${part_list}" ] && part_list="0,6,16,26,37"
# [ -z "${part_list}" ] && part_list="0,16,37"
[ -z "${part_list}" ] && part_list="0,37"

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='./hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12345
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
# [ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=-1

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
echo "RANK : ${RANK}"
echo "LOCAL_RANK : ${LOCAL_RANK}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"
echo "OMPI_COMM_WORLD_LOCAL_RANK: ${OMPI_COMM_WORLD_LOCAL_RANK}"

# echo "AZUREML_EXPERIMENT_ID: ${AZUREML_EXPERIMENT_ID}"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "n_layers: ${layers}"
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

wandb login --relogin $wandb_key
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

torchrun $DISTRIBUTED_ARGS sfm/tasks/progpt/ft_bfm_llama_inst.py \
          --encoder_attention_heads $num_head \
          --encoder_layers $layers \
          --encoder_ffn_embed_dim $ffn_size \
          --encoder_embed_dim $hidden_size \
          --droppath_prob $droppath_prob \
          --attn_dropout $attn_dropout \
          --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
          --sandwich_ln \
          --train_data_path $train_data_path \
          --valid_data_path $valid_data_path \
          --pipeline_model_parallel_size $pipeline_model_parallel_size \
          --tensor_model_parallel_size $tensor_model_parallel_size \
          --train_batch_size $train_batch_size \
          --val_batch_size $val_batch_size \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --seed 6666 \
          --ft \
          --bf16 \
          --d_tilde $d_tilde \
          --max_lr $max_lr \
          --save_dir $save_dir \
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --loadcheck_path $loadcheck_path \
          --llm_model_name_or_path $llm_model_name_or_path \
          --tokenizer_path $tokenizer_path \
          --pool_mode $pool_mode \
          --strategy $strategy \
          --embedding_length $embedding_length \
          --model_max_length $model_max_length \
          --pp_partition_layer_name $pp_partition_layer_name \
          --pp_part_list $part_list \
          --loadbfmckpt_path $loadbfmckpt_path \
          --log_interval $log_interval --load_ckpt \
          --save_batch_interval $save_batch_interval \
          --unfreeze_param_list "mol_adaptor,mol_rep_layernorm,decoder" \
          --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
          --finetune_from_checkpoint_dir $finetune_from_checkpoint_dir \
          --finetune_from_checkpoint_id $finetune_from_checkpoint_id \
          --instruction_mode
          # --ifresume


sleep inf
sleep inf
sleep inf
