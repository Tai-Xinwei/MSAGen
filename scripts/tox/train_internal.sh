#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set_before=$( set -o posix; set | sed -e '/^_=*/d' )


ulimit -c unlimited
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# ToxInternalLMDBDataset dataset args:
[ -z "${data_path}" ] && data_path="/mnta/yaosen/data/AFDB30-plddt70.lmdb"
[ -z "${max_length}" ] && max_length=128
[ -z "${min_length}" ] && min_length=20
[ -z "${num_residues}" ] && num_residues=32
[ -z "${transform_str}" ] && transform_str="[FromNumpy(), ItemToCRABBackBone(), CRABToInternal(), BERTMasking()]"

# ToxInternalModel args:
[ -z "${load_ckpt_from}" ] && load_ckpt_from=""

# model args
[ -z "${dropout}" ] && dropout=0.0
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${layerdrop}" ] && layerdrop=0.0
[ -z "${activation_dropout}" ] && activation_dropout=0.0
[ -z "${attntion_dropout}" ] && attntion_dropout=0.0
[ -z "${activation_fn}" ] && activation_fn="gelu"

[ -z "${num_encoder_layers}" ] && num_encoder_layers=1
[ -z "${embedding_dim}" ] && embedding_dim=256
[ -z "${ffn_embedding_dim}" ] && ffn_embedding_dim=256
[ -z "${num_attention_heads}" ] && num_attention_heads=8
[ -z "${q_noise}" ] && q_noise=0.0
[ -z "${qn_block_size}" ] && qn_block_size=8
[ -z "${d_tilde}" ] && d_tilde=1.0
[ -z "${add_rope}" ] && add_rope=false
[ -z "${export}" ] && export=false

# InitialLoss args
[ -z "${seq_type_loss_weight}" ] && seq_type_loss_weight=1.0
[ -z "${disto_loss_weight}" ] && disto_loss_weight=0.1
[ -z "${bl_loss_weight}" ] && bl_loss_weight=1.0
[ -z "${ba_loss_weight}" ] && ba_loss_weight=1.0
[ -z "${ba_norm_loss_weight}" ] && ba_norm_loss_weight=0.1
[ -z "${da_loss_weight}" ] && da_loss_weight=1.0
[ -z "${da_norm_loss_weight}" ] && da_norm_loss_weight=0.1
[ -z "${eps}" ] && eps=1e-5

# TrainerConfig args
[ -z "${seed}" ] && seed=666
[ -z "${fp16}" ] && fp16=false
[ -z "${auto_cast}" ] && auto_cast=false
[ -z "${bf16}" ] && bf16=false
[ -z "${grad_scaler_init}" ] && grad_scaler_init=1.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
# used in trainer, set it to a large number, we do length clip in dataset class with max_length
[ -z "${max_tokens}" ] && max_tokens=2048
[ -z "${train_batch_size}" ] && train_batch_size=32
[ -z "${val_batch_size}" ] && val_batch_size=32
# [ -z "${val_batch_interval}" ] && val_batch_interval=0
# [ -z "${val_batch_log_interval}" ] && val_batch_log_interval=1000
# [ -z "${val_epoch_interval}" ] && val_epoch_interval=1
[ -z "${save_dir}" ] && save_dir="./checkpoints"
[ -z "${save_batch_interval}" ] && save_batch_interval=0
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${log_interval}" ] && log_interval=10
[ -z "${strategy}" ] && strategy="Zero1"
# pp_partition_layer_name: str = ""
# pp_part_list: Optional[List[int]] = None
# [ -z "${cpu}" ] && cpu=false
# we do not use this parameter defined in trainer, see load_ckpt_from in ToxInternalModel
[ -z "${ifresume}" ] && ifresume=false
# [ -z "${load_ckpt}" ] && load_ckpt=false
# [ -z "${freeze_param_list}" ] && freeze_param_list=""
# [ -z "${unfreeze_param_list}" ] && unfreeze_param_list=""
# [ -z "${finetune_from_checkpoint_dir}" ] && finetune_from_checkpoint_dir=""
# [ -z "${finetune_from_checkpoint_id}" ] && finetune_from_checkpoint_id=""

# dataloader strategy
# daliLoader: bool = False
# dynamic_loader: bool = False
# ifstack: bool = False

# gradient_clipping: float = 1.0
[ -z "${total_num_steps}" ] && total_num_steps=1000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=60
# warmup_factor: float = 0.06
# warmup_lr: float = 1e-6
# warmup_num_epochs: int = 10
[ -z "${max_lr}" ] && max_lr=1e-4
# init_lr: float = 8e-5
# min_lr: float = 8e-6
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${total_num_epochs}" ] && total_num_epochs=100

# wandb
[ -z "${wandb}" ] && wandb=true
[ -z "${wandb_group}" ] && wandb_group=tox_dev
[ -z "${wandb_project}" ] && wandb_project=SFM_tox
[ -z "${wandb_team}" ] && wandb_team="yaosenmin"
[ -z "${wandb_key}" ] && wandb_key=local-138548ae9c9a3b39646af8ae2c4c6d4e22c51385

# adam
# beta1: float = 0.9
# beta2: float = 0.999
# eps: float = 1e-8

# early stopping
# early_stopping: bool = False
# early_stopping_patience: int = 10
# early_stopping_metric: str = "valid_loss"
# early_stopping_mode: str = "min"

# compile CUDA kernels with torch.compile
# compile: bool = False

# validate
# calculate_metrics: bool = False

action_args=""
if [ "$wandb" = "true" ]; then
  action_args+="--wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project "
  wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key
  export WANDB_API_KEY=$wandb_key
fi

if [ "$add_rope" = "true" ]; then action_args+="--add_rope "; fi
if [ "$export" = "true" ]; then action_args+="--export "; fi
if [ "$fp16" = "true" ]; then action_args+="--fp16 "; fi
if [ "$auto_cast" = "true" ]; then action_args+="--auto_cast "; fi
if [ "$bf16" = "true" ]; then action_args+="--bf16 "; fi
if [ "$ifresume" = "true" ]; then action_args+="--ifresume "; fi


[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

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

echo -e "\n\n"
echo "=====================================ARGS======================================"
set_after=$( set -o posix; unset set_before; set | sed -e '/^_=/d' )
diff  <(echo "$set_before") <(echo "$set_after") | sed -e 's/^> //' -e '/^[[:digit:]].*/d'
# hack from https://stackoverflow.com/questions/1305237/how-to-list-variables-declared-in-script-in-bash

echo "==================================ACTION ARGS==========================================="
echo "action_args: ${action_args}"
echo "========================================================================================"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8

DISTRIBUTED_ARGS=""
if (( $OMPI_COMM_WORLD_SIZE == 1 )); then
  DISTRIBUTED_ARGS="--nproc_per_node $n_gpu --master_port $MASTER_PORT"
else
  DISTRIBUTED_ARGS="--nproc_per_node $n_gpu --nnodes $OMPI_COMM_WORLD_SIZE --node_rank $OMPI_COMM_WORLD_RANK --master_addr $MASTER_ADDR"
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"

torchrun $DISTRIBUTED_ARGS sfm/tasks/tox/train_internal.py \
    --data_path $data_path \
    --seed $seed \
    --max_length $max_length \
    --min_length $min_length \
    --num_residues $num_residues \
    --transform_str "$transform_str" \
    --dropout $dropout \
    --droppath_prob $droppath_prob \
    --layerdrop $layerdrop \
    --activation_dropout $activation_dropout \
    --attntion_dropout $attntion_dropout \
    --activation_fn $activation_fn \
    --num_encoder_layers $num_encoder_layers \
    --embedding_dim $embedding_dim \
    --ffn_embedding_dim $ffn_embedding_dim \
    --num_attention_heads $num_attention_heads \
    --q_noise $q_noise \
    --qn_block_size $qn_block_size \
    --d_tilde $d_tilde \
    --seq_type_loss_weight $seq_type_loss_weight \
    --disto_loss_weight $disto_loss_weight \
    --bl_loss_weight $bl_loss_weight \
    --ba_loss_weight $ba_loss_weight \
    --ba_norm_loss_weight $ba_norm_loss_weight \
    --da_loss_weight $da_loss_weight \
    --da_norm_loss_weight $da_norm_loss_weight \
    --eps $eps \
    --grad_scaler_init  $grad_scaler_init \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_tokens $max_tokens \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --save_dir $save_dir \
    --save_batch_interval $save_batch_interval \
    --save_epoch_interval $save_epoch_interval \
    --log_interval $log_interval \
    --strategy $strategy \
    --total_num_steps $total_num_steps \
    --warmup_num_steps $warmup_num_steps \
    --max_lr $max_lr \
    --weight_decay $weight_decay \
    --total_num_epochs $total_num_epochs \
    $action_args
