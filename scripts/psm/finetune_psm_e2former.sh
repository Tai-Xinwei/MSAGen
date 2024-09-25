#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${backbone_config}" ] && backbone_config='e2former'
[ -z "${backbone}" ] && backbone='e2former'
[ -z "${psm_finetune_mode}" ] && psm_finetune_mode=true

[ -z "${layers}" ] && layers=8
[ -z "${hidden_size}" ] && hidden_size=128
[ -z "${ffn_size}" ] && ffn_size=4096
[ -z "${num_head}" ] && num_head=8
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=2
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${loss_unit}" ] && loss_unit="kcal/mol"
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=512
# [ -z "${max_tokens}" ] && max_tokens=24000
[ -z "${max_tokens}" ] && max_tokens=36000

[ -z "${dropout}" ] && dropout=0.1
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${noise_mode}" ] && noise_mode=diff
[ -z "${psm_finetune_valid_noise_mode}" ] && psm_finetune_valid_noise_mode=zero
[ -z "${psm_finetune_noise_mode}" ] && psm_finetune_noise_mode=zero
[ -z "${node_type_edge_method}" ] && node_type_edge_method=EXCHANGABLE

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=true
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=true

[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=2e-4
[ -z "${total_num_steps}" ] && total_num_steps=200000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=1000
[ -z "${train_batch_size}" ] && train_batch_size=512
[ -z "${val_batch_size}" ] && val_batch_size=512
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=8
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=10000000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=3000
[ -z "${num_timesteps}" ] && num_timesteps=5000

[ -z "${mode_prob}" ] && mode_prob='0.1,0.2,0.6,0.1' #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full
# [ -z "${mode_prob}" ] && mode_prob='0.0,0.0,0.0,1.0' # prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full

# [ -z "${data_path}" ] && data_path='/fastdata/peiran/tox/48organisms-fullatom.lmdb/'

[ -z "${data_path}" ] && data_path="/home/hul/wangchu/data/"
# [ -z "${data_path_list}" ] && data_path_list="SPICE-2.0.1/SPICE_PubChem_Set_1_Single_Points_Dataset_v1.3"
# [ -z "${dataset_name_list}" ] && dataset_name_list="SPICE-2.0.1/SPICE_PubChem_Set_1_Single_Points_Dataset_v1.3"
[ -z "${data_path_list}" ] && data_path_list="deshaw-filter/"
[ -z "${shuffle}" ] && shuffle=True
[ -z "${dataset_name_list}" ] && dataset_name_list="deshaw"
[ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
[ -z "${loadcheck_path}" ] && loadcheck_path=''
[ -z "${save_dir}" ] && save_dir='/data/SPICE'
[ -z "${dataset_name}" ] && dataset_name="."
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0

[ -z "${wandb}" ] && wandb=false
[ -z "${wandb_group}" ] && wandb_group=""
[ -z "${wandb_team}" ] && wandb_team=faralley
[ -z "${wandb_project}" ] && wandb_project=psm_debug_workshop
[ -z "${wandb_run_name}" ] && wandb_run_name=SPICE
[ -z "${wandb_key}" ] && wandb_key=1059e2793fc0c6ba4d85481bb10d9d1930e34ef1
# [ -z "${wandb_group}" ] && wandb_group=psm_dev
# [ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
# [ -z "${wandb_project}" ] && wandb_project=psm_dev
# [ -z "${wandb_key}" ] && wandb_key=local-094f941ede8eda7a00c307f50595f054be5382f7



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

# wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key

# wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key
wandb login --relogin --host=https://api.wandb.ai $wandb_key

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

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"


torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/finetune_psm.py \
          --config-name=config_psm.yaml \
          backbone_config=$backbone_config \
          backbone=$backbone \
          psm_finetune_mode=$psm_finetune_mode \
          loss_unit=$loss_unit \
          encoder_embed_dim=$hidden_size \
          backbone_config.num_layers=$layers \
          encoder_layers=$layers \
          backbone_config.max_radius=5 \
          droppath_prob=$droppath_prob \
          clean_sample_ratio=1.0 \
          attn_dropout=$attn_dropout \
          act_dropout=$act_dropout \
          dropout=$dropout \
          weight_decay=$weight_decay \
          sandwich_ln=True \
          add_3d=True \
          data_path=$data_path \
          data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
          dataset_split_raito=\"$dataset_split_raito\" \
          shuffle=$shuffle \
          save_dir=$save_dir \
          seed=12345 \
          ifresume=False \
          mask_ratio=$mask_ratio \
          noise_scale=$noise_scale \
          num_pred_attn_layer=$num_pred_attn_layer \
          d_tilde=$d_tilde \
          strategy=$strategy \
          max_lr=$max_lr \
          num_timesteps=$num_timesteps \
          mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
          psm_finetune_valid_noise_mode=$psm_finetune_valid_noise_mode \
          use_2d_atom_features=False use_2d_bond_features=False \
          total_num_steps=$total_num_steps \
          warmup_num_steps=$warmup_num_steps \
          train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
          save_batch_interval=$save_batch_interval log_interval=$log_interval \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project wandb_run_name=$wandb_run_name \
