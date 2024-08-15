#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'


# wget 'https://aka.ms/downloadazcopy-v10-linux' -O /tmp/azcopy.tar.gz
# tar -xf /tmp/azcopy.tar.gz -C /tmp
# # find the folder in /tmp and starts with azcopy_linux_amd64
# azcopy_path=$(find /tmp -maxdepth 1 -type d -name 'azcopy_linux_amd64*')
# # $azcopy_path/azcopy copy ... ... --recursive


[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=1024
[ -z "${ffn_size}" ] && ffn_size=4096
[ -z "${num_head}" ] && num_head=32
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=2
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=512
[ -z "${max_tokens}" ] && max_tokens=2000
# [ -z "${max_tokens}" ] && max_tokens=36000

[ -z "${dropout}" ] && dropout=0.1
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_mode}" ] && noise_mode=diff

[ -z "${mask_ratio}" ] && mask_ratio=0.0
[ -z "${clean_sample_ratio}" ] && clean_sample_ratio=0.0

[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${total_num_steps}" ] && total_num_steps=1
[ -z "${warmup_num_steps}" ] && warmup_num_steps=1
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=2000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=1
[ -z "${val_batch_interval}" ] && val_batch_interval=10000
[ -z "${mode_prob}" ] && mode_prob='0.0,1.0,0.0'

[ -z "${data_path}" ] && data_path='/fastdata/peiran/psm/'
# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="80"
# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,20240630_PDB_Training_Data,AFDB50-plddt70.lmdb'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6,pdbcomplexmultimer,afdb'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.5,0.1,0.4'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="112,24,24"
# [ -z "${data_path_list}" ] && data_path_list='20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='12'
# [ -z "${data_path_list}" ] && data_path_list='matter-sim-15M-merged,'
# [ -z "${dataset_name_list}" ] && dataset_name_list='mattersim'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="12"
[ -z "${data_path_list}" ] && data_path_list='20240630_PDB_Training_Data'
[ -z "${dataset_name_list}" ] && dataset_name_list='pdbcomplexmultimer'
[ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='1'
# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,matter-sim-15M-force-filtered-merged,AFDB50-plddt70.lmdb,matter-sim-15M-merged,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6,mattersim,afdb,mattersim,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.3,0.05,0.45,0.15,0.05'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='80,12,24,12,8'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='112,12,24,12,24'
# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,AFDB50-plddt70.lmdb,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6,afdb,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.5,0.4,0.1'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='200,32,48,32'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='112,24,24'
# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,matter-sim-15M-force-filtered-merged,AFDB50-plddt70.lmdb,matter-sim-15M-merged,ur50_23_bpe_pack1536.lmdb,20240101_PDB_Training_Data,complex.preprocessed.large'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6,mattersim,afdb,mattersim,ur50,pdb,complex'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.2,0.1,0.3,0.1,0.1,0.1,0.1'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='128,12,32,12,16,32,32'

[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${AutoGradForce}" ] && AutoGradForce=False
[ -z "${NoisePredForce}" ] && NoisePredForce=False
[ -z "${force_head_type}" ] && force_head_type=MLP
[ -z "${force_loss_type}" ] && force_loss_type=L1
[ -z "${molecule_energy_loss_ratio}" ] && molecule_energy_loss_ratio=1.0
[ -z "${material_energy_loss_ratio}" ] && material_energy_loss_ratio=1.0
[ -z "${material_force_loss_ratio}" ] && material_force_loss_ratio=1.0
[ -z "${energy_per_atom_label_scale}" ] && energy_per_atom_label_scale=1.0

[ -z "${rescale_loss_with_std}" ] && rescale_loss_with_std=True

[ -z "${use_dali_pipeline}" ] && use_dali_pipeline=False
[ -z "${fp16}" ] && fp16=False
[ -z "${mm_tensorcore}" ] && mm_tensorcore="tf32"
[ -z "${compile}" ] && compile=False

[ -z "${loadcheck_path}" ] && loadcheck_path="/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_dit_v13_300m/checkpoints/global_step88639/mp_rank_00_model_states.pt"
[ -z "${finetune_from_checkpoint_id}" ] && finetune_from_checkpoint_id="global_step252285"
[ -z "${save_dir}" ] && save_dir='/data/peiran/output/'

[ -z "${wandb_group}" ] && wandb_group=psm_dev_vt
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=psm_dev
[ -z "${wandb_key}" ] && wandb_key=local-094f941ede8eda7a00c307f50595f054be5382f7

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="RELATIVE_POS_VEC_BIAS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=20.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=256
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=10.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=True

[ -z "${diffusion_noise_std}" ] && diffusion_noise_std=10.0
[ -z "${diffusion_mode}" ] && diffusion_mode=epsilon
[ -z "${diff_init_lattice_size}" ] && diff_init_lattice_size=10.0
[ -z "${diffusion_sampling}" ] && diffusion_sampling="ode"
[ -z "${num_timesteps_stepsize}" ] && num_timesteps_stepsize=-10
[ -z "${diffusion_training_loss}" ] && diffusion_training_loss="L1"
[ -z "${sample_ligand_only}" ] && sample_ligand_only=True
[ -z "${max_residue_num}" ] && max_residue_num=2048
[ -z "${crop_radius}" ] && crop_radius=100

[ -z "${num_timesteps}" ] && num_timesteps=5000
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=1e-7
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=2e-3
[ -z "${ddpm_schedule}" ] && ddpm_schedule=sigmoid

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=False
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=False
[ -z "${use_smooth_equviariant_norm}" ] && use_smooth_equviariant_norm=True
[ -z "${num_edges}" ] && num_edges=25600
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=32

[ -z "${use_2d_atom_features}" ] && use_2d_atom_features=True
[ -z "${use_2d_bond_features}" ] && use_2d_bond_features=False
[ -z "${only_use_rotary_embedding_for_protein}" ] && only_use_rotary_embedding_for_protein=True
[ -z "${use_hard_dist_loss}" ] && use_hard_dist_loss=True
[ -z "${if_total_energy}" ] && if_total_energy=False
[ -z "${decoder_feat4energy}" ] && decoder_feat4energy=False
[ -z "${disable_data_aug}" ] && disable_data_aug=False

[ -z "${psm_validation_mode}" ] && psm_validation_mode=True
[ -z "${sample_in_validation}" ] && sample_in_validation=True
[ -z "${num_sampling_time}" ] && num_sampling_time=1
# [ -z "${sampled_structure_output_path}" ] && sampled_structure_output_path="../sample_save_dir_complex"
[ -z "${sampled_structure_output_path}" ] && sampled_structure_output_path="../sample_save_dir_complex_posebuster"

[ -z "${psm_finetune_mode}" ] && psm_finetune_mode=True
[ -z "${psm_sample_structure_in_finetune}" ] && psm_sample_structure_in_finetune=False
[ -z "${psm_finetune_reset_head}" ] && psm_finetune_reset_head=False
[ -z "${val_batch_log_all_metric}" ] && val_batch_log_all_metric=False
[ -z "${psm_validate_for_train_set}" ] && psm_validate_for_train_set=False
[ -z "${val_batch_log_interval}" ] && val_batch_log_interval=1

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
echo "dataset_name: ${dataset_name}"
echo "mask_ratio: ${mask_ratio}"
echo "mode_prob: ${mode_prob}"
echo "noise_mode: ${noise_mode}"

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
          backbone_config=graphormer \
          backbone=dit \
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
          data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
          dataset_split_raito=\"$dataset_split_raito\" \
          save_dir=$save_dir \
          seed=1666 \
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
          psm_validation_mode=$psm_validation_mode num_edges=$num_edges num_3d_bias_kernel=$num_3d_bias_kernel \
          diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
          num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
          ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          clean_sample_ratio=$clean_sample_ratio \
          use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
          use_dali_pipeline=$use_dali_pipeline molecule_energy_loss_ratio=$molecule_energy_loss_ratio \
          material_energy_loss_ratio=$material_energy_loss_ratio material_force_loss_ratio=$material_force_loss_ratio \
          energy_per_atom_label_scale=$energy_per_atom_label_scale molecule_energy_per_atom_std_override=1.0 \
          preprocess_2d_bond_features_with_cuda=True use_smooth_equviariant_norm=$use_smooth_equviariant_norm \
          AutoGradForce=$AutoGradForce force_head_type=$force_head_type psm_finetune_mode=$psm_finetune_mode \
          only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
          diffusion_training_loss=$diffusion_training_loss use_hard_dist_loss=$use_hard_dist_loss \
          mm_tensorcore=$mm_tensorcore compile=$compile disable_data_aug=$disable_data_aug \
          if_total_energy=$if_total_energy decoder_feat4energy=$decoder_feat4energy \
          NoisePredForce=$NoisePredForce force_loss_type=$force_loss_type \
          rescale_loss_with_std=$rescale_loss_with_std \
          loadcheck_path=$loadcheck_path \
          sample_in_validation=$sample_in_validation \
          sampled_structure_output_path=$sampled_structure_output_path \
          num_sampling_time=$num_sampling_time \
          psm_sample_structure_in_finetune=$psm_sample_structure_in_finetune \
          psm_finetune_reset_head=$psm_finetune_reset_head \
          val_batch_log_all_metric=$val_batch_log_all_metric \
          psm_validate_for_train_set=$psm_validate_for_train_set \
          val_batch_log_interval=$val_batch_log_interval \
          num_timesteps_stepsize=$num_timesteps_stepsize \
          sample_ligand_only=$sample_ligand_only \
          max_residue_num=$max_residue_num crop_radius=$crop_radius \

sleep infinity
