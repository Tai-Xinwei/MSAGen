#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

echo 'Solving MKL done!'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=3072
[ -z "${num_head}" ] && num_head=8
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=2
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=256
[ -z "${max_tokens}" ] && max_tokens=3000
[ -z "${num_timesteps}"] && num_timesteps=1000

[ -z "${dropout}" ] && dropout=0.1
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${noise_mode}" ] && noise_mode=diff
[ -z "${lamb_ism}" ] && lamb_ism=0.01
[ -z "${lamb_pde_q}" ] && lamb_pde_q=0.001
[ -z "${lamb_pde_control}" ] && lamb_pde_control=0.001
[ -z "${diffmode}" ] && diffmode=score
# [ -z "${seq_masking_method}" ] && seq_masking_method=continuousMask
[ -z "${seq_masking_method}" ] && seq_masking_method=transformerM

[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${total_num_steps}" ] && total_num_steps=2000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=1000
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=1000

[ -z "${mode_prob}" ] && mode_prob='0.1,0.2,0.6,0.1' #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full
# [ -z "${mode_prob}" ] && mode_prob='0.0,1.0,0.0,0.0' # prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full

# [ -z "${data_path}" ] && data_path='/fastdata/peiran/bfm/48organism1m.lmdb/'
# [ -z "${data_path}" ] && data_path='/mnt/protein/48organism1m.lmdb'
[ -z "${data_path}" ] && data_path='/blob/hai1data/pfm/data/afdb/48organisms-fullatom.lmdb/'
# [ -z "${loadcheck_path}" ] && loadcheck_path='/fastdata/peiran/tox/checkpoints/pfmdiff100M768_prob1522_m5_bs256_ddpmnoise_v1_pi_dist_score/'
[ -z "${save_dir}" ] && save_dir='/fastdata/peiran/tox/checkpoints/pfmdiff100M768_prob1522_m5_bs256_ddpmnoise_v1_pi_dist_score/'
[ -z "${loadcheck_path}" ] && loadcheck_path='.'
[ -z "${diffmode}" ] && diffmode='ddpm'

# [ -z "${save_dir}" ] && save_dir='/home/peiran/FMproj/output/'
[ -z "${dataset_name}" ] && dataset_name="."
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0

[ -z "${wandb_group}" ] && wandb_group=bfmdiff
[ -z "${wandb_team}" ] && wandb_team=peiranjin
[ -z "${wandb_project}" ] && wandb_project=ds_mfmpre

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
echo "mode_prob: ${mode_prob}"
echo "noise_mode: ${noise_mode}"
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1

# wandb login --relogin 680f261d2b178f36d57f4644e235d9db1c207bc0
# export WANDB_API_KEY=680f261d2b178f36d57f4644e235d9db1c207bc0


if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_nod $n_gpu \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
  fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"

# CUDA_VISIBLE_DEVICE=0
torchrun $DISTRIBUTED_ARGS sfm/tasks/tox/sample_tox.py \
          --encoder_attention_heads $num_head \
          --encoder_layers $layers \
          --encoder_ffn_embed_dim $ffn_size \
          --encoder_embed_dim $hidden_size \
          --droppath_prob $droppath_prob \
          --attn_dropout $attn_dropout \
          --act_dropout $act_dropout --dropout $dropout --weight_decay $weight_decay \
          $("$sandwich_ln" == "true" && echo "--sandwich_ln") \
          --dataset_names $dataset_name \
          --data_path $data_path \
          --save_dir $save_dir \
          --seed 666666 \
          $("$add_3d" == "true" && echo "--add_3d") \
          $("$no_2d" == "true" && echo "--no_2d") \
          --mask_ratio $mask_ratio \
          --noise_scale $noise_scale \
          --num_pred_attn_layer $num_pred_attn_layer \
          --d_tilde $d_tilde \
          --lamb_ism $lamb_ism \
          --lamb_pde_q $lamb_pde_q \
          --lamb_pde_control $lamb_pde_control \
          --diffmode $diffmode \
          --strategy $strategy \
          --max_lr $max_lr \
          --num_timesteps $num_timesteps \
          --seq_masking_method $seq_masking_method \
          --mode_prob $mode_prob --noise_mode $noise_mode\
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --train_batch_size $train_batch_size --val_batch_size $val_batch_size --max_length $max_length \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --save_epoch_interval $save_epoch_interval --total_num_epochs $epochs \
          --save_batch_interval $save_batch_interval --log_interval $log_interval \
          --wandb --wandb_group $wandb_group --wandb_team $wandb_team --wandb_project $wandb_project \
          --sample_mode \
          --infer \
          --loadcheck_path $loadcheck_path \
          --diffmode $diffmode \
          $("$ode_mode" == "true" && echo "--ode_mode") \


# sleep inf
# sleep inf
# sleep inf
