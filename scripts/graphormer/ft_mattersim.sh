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

[ -z "${pbc_cutoff}" ] && pbc_cutoff=20.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=256
[ -z "${force_loss_factor}" ] && force_loss_factor=1.0

[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=2e-4
[ -z "${total_num_steps}" ] && total_num_steps=5000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=30000
[ -z "${train_batch_size}" ] && train_batch_size=32
[ -z "${val_batch_size}" ] && val_batch_size=32
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${save_batch_interval}" ] && save_batch_interval=20000
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=100

[ -z "${data_path}" ] && data_path='/mntd/shiyu/dataset/matter-sim-15M' #pubchem-pm6-small'
[ -z "${loadcheck_path}" ] && loadcheck_path="" #"/mntd/mattersim/checkpoints/checkpoint_E5_B1899.pt"
[ -z "${save_dir}" ] && save_dir='/mntd/shiyu/checkpoints/matter-sim-debug/'
[ -z "${dataset_names}" ] && dataset_names="PM6-Full-3D"
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${strategy}" ] && strategy=DDP

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62348
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
echo "save_dir: ${save_dir}"
echo "dataset_name: ${dataset_name}"
echo "noise_scale: ${noise_scale}"
echo "mask_ratio: ${mask_ratio}"



# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1


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

if [[ $OMPI_COMM_WORLD_SIZE > 1 ]]; then
  cp sfm/utils/barrier.py . && touch READY && python -u barrier.py $OMPI_COMM_WORLD_SIZE $OMPI_COMM_WORLD_RANK
fi

torchrun ${DISTRIBUTED_ARGS} sfm/tasks/graphormer/ft_mattersim.py \
          --num_classes 1 \
          --encoder_attention_heads $num_head \
          --encoder_layers $layers \
          --encoder_ffn_embed_dim $ffn_size \
          --encoder_embed_dim $hidden_size \
          --droppath_prob $droppath_prob \
          --attn_dropout $attn_dropout \
          --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
          --sandwich_ln \
          --dataset_names $dataset_names \
          --data_path $data_path \
          --save_dir $save_dir \
          --seed 12345 \
          --add_3d --ft \
          --d_tilde $d_tilde \
          --num_pred_attn_layer $num_pred_attn_layer \
          --max_lr $max_lr \
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --loadcheck_path "${loadcheck_path}" \
          --strategy $strategy \
          --train_batch_size $train_batch_size --val_batch_size $val_batch_size \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --save_epoch_interval $save_epoch_interval --total_num_epochs $epochs \
          --save_batch_interval $save_batch_interval \
          --log_interval $log_interval \
          --use_pbc \
          --num_edges 16384 \
          --pbc_cutoff ${pbc_cutoff} \
          --pbc_expanded_num_cell_per_direction ${pbc_expanded_num_cell_per_direction} \
          --pbc_expanded_token_cutoff ${pbc_expanded_token_cutoff} \
          --no_2d \
          --force_loss_factor ${force_loss_factor} \
          --use_2d_atom_features \
          --use_2d_bond_features \
          --diffusion_training_loss L1 \
          --clean_sample_ratio 0.5
