#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=32
[ -z "${hidden_size}" ] && hidden_size=1536
[ -z "${ffn_size}" ] && ffn_size=6144
[ -z "${num_head}" ] && num_head=32
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=512
[ -z "${max_tokens}" ] && max_tokens=2000
# [ -z "${max_tokens}" ] && max_tokens=36000

[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.0
[ -z "${attn_dropout}" ] && attn_dropout=0.0
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_mode}" ] && noise_mode=diff

[ -z "${mask_ratio}" ] && mask_ratio=0.0
[ -z "${clean_sample_ratio}" ] && clean_sample_ratio=1.0
[ -z "${psm_finetune_noise_mode}" ] && psm_finetune_noise_mode=T
[ -z "${finetune_module}" ] && finetune_module=plddt_confidence_head
[ -z "${psm_sample_structure_in_finetune}" ] && psm_sample_structure_in_finetune=True

[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=4e-5
[ -z "${total_num_steps}" ] && total_num_steps=50000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=10
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=0
[ -z "${save_batch_interval}" ] && save_batch_interval=10000
[ -z "${log_interval}" ] && log_interval=1
[ -z "${epochs}" ] && epochs=1000
[ -z "${val_batch_interval}" ] && val_batch_interval=10000
[ -z "${mode_prob}" ] && mode_prob='0.0,1.0,0.0'

[ -z "${data_path}" ] && data_path=~/
[ -z "${data_path_list}" ] && data_path_list='20240101_PDB_Training_Data' # 'AFDB30-plddt90.lmdb' #
# [ -z "${data_path}" ] && data_path='/fastdata/peiran/psm/'
# [ -z "${data_path_list}" ] && data_path_list='AFDB50-plddt70.lmdb'
[ -z "${dataset_name_list}" ] && dataset_name_list='pdb' # 'afdb' #
[ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="1"
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${use_dali_pipeline}" ] && use_dali_pipeline=False
[ -z "${fp16}" ] && fp16=False
[ -z "${energy_loss_ratio}" ] && energy_loss_ratio=1.0
[ -z "${force_loss_ratio}" ] && force_loss_ratio=1.0

[ -z "${loadcheck_path}" ] && loadcheck_path="/sfmdataeastus2/psm/exp/peiran/psmv1_vt_v10_1b/checkpoints/global_step200000/mp_rank_00_model_states.pt"
# [ -z "${loadcheck_path}" ] && loadcheck_path="/data/peiran/blob/sfmdataeastus2/psm/exp/peiran/psmv1_vt_v10_1b/checkpoints/global_step200000/mp_rank_00_model_states.pt"
[ -z "${save_dir}" ] && save_dir='home/peiranjin/output'

[ -z "${wandb_project}" ] && wandb_project=psm_VT_finetune
[ -z "${wandb_group}" ] && wandb_group=protein_confidence_model_plddt_finetune
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_run_name}" ] && wandb_run_name=protein_confidence_model_plddt_finetune$(date -u +%Y%m%d.%H%M%S)
[ -z "${wandb_key}" ] && wandb_key="local-138548ae9c9a3b39646af8ae2c4c6d4e22c51385"

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="RELATIVE_POS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=20.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=3
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=256
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=5.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=False
[ -z "${diffusion_noise_std}" ] && diffusion_noise_std=10.0
[ -z "${diffusion_mode}" ] && diffusion_mode=epsilon

[ -z "${diff_init_lattice_size}" ] && diff_init_lattice_size=10.0
[ -z "${diffusion_sampling}" ] && diffusion_sampling="ode"
[ -z "${num_timesteps}" ] && num_timesteps=5000
[ -z "${num_timesteps_stepsize}" ] && num_timesteps_stepsize=25
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=1e-7
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=2e-3
[ -z "${ddpm_schedule}" ] && ddpm_schedule=sigmoid

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=False
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=False
[ -z "${psm_validation_mode}" ] && psm_validation_mode=False
[ -z "${use_2d_atom_features}" ] && use_2d_atom_features=True
[ -z "${use_2d_bond_features}" ] && use_2d_bond_features=False

[ -z "${freeze_backbone}" ] && freeze_backbone=True
[ -z "${force_head_type}" ] && force_head_type=GATED_EQUIVARIANT

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
echo "data_path: ${data_path}"
echo "output_path: ${output_path}"
echo "mask_ratio: ${mask_ratio}"
echo "mode_prob: ${mode_prob}"
echo "noise_mode: ${noise_mode}"
echo "freeze_backbone: ${freeze_backbone}"

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=4

wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key
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

torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
          --config-name=config_psm.yaml \
          psm_finetune_mode=True \
          psm_finetune_noise_mode=$psm_finetune_noise_mode psm_finetune_valid_noise_mode=T \
          finetune_module=$finetune_module \
          backbone_config=graphormer \
          backbone=vanillatransformer \
          encoder_attention_heads=$num_head \
          encoder_layers=$layers \
          num_pred_attn_layer=$num_pred_attn_layer \
          encoder_ffn_embed_dim=$ffn_size \
          encoder_embed_dim=$hidden_size \
          droppath_prob=$droppath_prob \
          attn_dropout=$attn_dropout \
          act_dropout=$act_dropout \
          dropout=$dropout \
          weight_decay=$weight_decay \
          sandwich_ln=True \
          data_path=$data_path \
          data_path_list=\"$data_path_list\" \
          dataset_split_raito=\"$dataset_split_raito\" dataset_name_list=\"$dataset_name_list\" \
          save_dir=$save_dir \
          seed=12345 \
          mask_ratio=$mask_ratio \
          d_tilde=$d_tilde \
          strategy=$strategy \
          max_lr=$max_lr \
          diffusion_mode=\"$diffusion_mode\" \
          mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
          total_num_steps=$total_num_steps \
          warmup_num_steps=$warmup_num_steps \
          train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
          save_batch_interval=$save_batch_interval log_interval=$log_interval \
          equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
          pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
          pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
          diffusion_noise_std=$diffusion_noise_std fp16=$fp16 \
          psm_validation_mode=$psm_validation_mode \
          diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
          num_timesteps=$num_timesteps num_timesteps_stepsize=$num_timesteps_stepsize ddpm_beta_start=$ddpm_beta_start \
          ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          clean_sample_ratio=$clean_sample_ratio psm_sample_structure_in_finetune=$psm_sample_structure_in_finetune \
          use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project wandb_run_name=$wandb_run_name \
          use_dali_pipeline=$use_dali_pipeline \
          energy_loss_ratio=$energy_loss_ratio force_loss_ratio=$force_loss_ratio \
          preprocess_2d_bond_features_with_cuda=True \
          loadcheck_path=$loadcheck_path \
          freeze_backbone=$freeze_backbone force_head_type=$force_head_type
