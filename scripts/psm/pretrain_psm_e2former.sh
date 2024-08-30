#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'
[ -z "${layers}" ] && layers=1
[ -z "${hidden_size}" ] && hidden_size=1024
[ -z "${ffn_size}" ] && ffn_size=4096
[ -z "${num_head}" ] && num_head=32
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=512
[ -z "${max_tokens}" ] && max_tokens=2000

[ -z "${dropout}" ] && dropout=0.1
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_mode}" ] && noise_mode=diff

[ -z "${mask_ratio}" ] && mask_ratio=0.3
[ -z "${clean_sample_ratio}" ] && clean_sample_ratio=0.5

[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1e-4
[ -z "${total_num_steps}" ] && total_num_steps=2000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=10000
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=8
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=20000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=1000
[ -z "${val_batch_interval}" ] && val_batch_interval=10000
[ -z "${mode_prob}" ] && mode_prob='0.2,0.6,0.2'
[ -z "${complex_mode_prob}" ] && complex_mode_prob='0.4,0.4,0.2' #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full
# [ -z "${mode_prob}" ] && mode_prob='0.0,0.0,0.0,1.0' # prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full



# [ -z "${data_path}" ] && data_path="/home/weixinran/data/sfm"
# export data_path_list='matter-sim-15M-force-filtered-merged,matter-sim-15M-merged,pm6_10M_refined4.lmdb,48organisms-fullatom.lmdb,20240630_PDB_Training_Data'
# export dataset_name_list='mattersim,mattersim,pm6,afdb,pdbcomplexmultimer'
# export dataset_split_raito='0.05,0.15,0.3,0.45,0.05'
# export dataset_micro_batch_size='8,8,32,8,2'

[ -z "${data_path}" ] && data_path="/home/weixinran/data/sfm"
[ -z "${data_path_list}" ] && data_path_list="pm6_10M_refined4.lmdb"
[ -z "${dataset_name_list}" ] && dataset_name_list="pm6" #"afdb"
[ -z "${dataset_split_raito}" ] && dataset_split_raito='1'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="32"

# [ -z "${data_path}" ] && data_path="/home/weixinran/data/sfm"
# [ -z "${data_path_list}" ] && data_path_list="AFDB50-plddt70.lmdb"
# [ -z "${dataset_name_list}" ] && dataset_name_list="afdb" #"afdb"
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="8"


# [ -z "${data_path}" ] && data_path="/home/weixinran/data/sfm"
# [ -z "${data_path_list}" ] && data_path_list="matter-sim-15M-merged"
# [ -z "${dataset_name_list}" ] && dataset_name_list="mattersim" #"afdb"
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="32"

# [ -z "${data_path}" ] && data_path="/home/weixinran/data/sfm"
# [ -z "${data_path_list}" ] && data_path_list="matter-sim-15M-force-filtered-merged"
# [ -z "${dataset_name_list}" ] && dataset_name_list="mattersim" #"afdb"
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="32"



# [ -z "${data_path}" ] && data_path="/home/weixinran/data/sfm"
# [ -z "${data_path_list}" ] && data_path_list="48organisms-fullatom.lmdb"
# [ -z "${dataset_name_list}" ] && dataset_name_list="afdb" #"afdb"
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="8"

[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${AutoGradForce}" ] && AutoGradForce=False
[ -z "${NoisePredForce}" ] && NoisePredForce=False
[ -z "${node_type_edge_method}" ] && node_type_edge_method=EXCHANGABLE
[ -z "${force_head_type}" ] && force_head_type=MLP
[ -z "${force_loss_type}" ] && force_loss_type=L1

[ -z "${molecule_energy_loss_ratio}" ] && molecule_energy_loss_ratio=1.0
[ -z "${material_energy_loss_ratio}" ] && material_energy_loss_ratio=1.0
[ -z "${material_force_loss_ratio}" ] && material_force_loss_ratio=1.0
[ -z "${energy_per_atom_label_scale}" ] && energy_per_atom_label_scale=1.0
[ -z "${molecule_ref_energy_source}" ] && molecule_ref_energy_source="PubChemQC-B3LYP-PM6/wb97xd3/1.0.0/train"
[ -z "${molecule_outlier_energy_atoms}" ] && molecule_outlier_energy_atoms=""

[ -z "${rescale_loss_with_std}" ] && rescale_loss_with_std=True

[ -z "${use_dali_pipeline}" ] && use_dali_pipeline=False
[ -z "${fp16}" ] && fp16=False
[ -z "${mm_tensorcore}" ] && mm_tensorcore="tf32"
[ -z "${compile}" ] && compile=False



[ -z "${loadcheck_path}" ] && loadcheck_path="/data/peiran/ckpt/psmv1_vt_v10_1b/global_step185000/mp_rank_00_model_states.pt"
[ -z "${finetune_from_checkpoint_id}" ] && finetune_from_checkpoint_id="global_step165000"
[ -z "${save_dir}" ] && save_dir='/data/peiran/output/'

# [ -z "${save_dir}" ] && save_dir='/home/peiran/FMproj/output/'

[ -z "${wandb_group}" ] && wandb_group=Lin_psm_dev_3mod
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=psm_dev
[ -z "${wandb_key}" ] && wandb_key=local-065f023e262b3ae11107532ba5463cd2d800d739
[ -z "${wandb_run_name}" ] && wandb_run_name=test

[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0


[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="RELATIVE_POS_VEC_BIAS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=7.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=256
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=10.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=True
[ -z "${diffusion_noise_std}" ] && diffusion_noise_std=10.0
[ -z "${diffusion_mode}" ] && diffusion_mode=epsilon


[ -z "${diff_init_lattice_size}" ] && diff_init_lattice_size=10.0
[ -z "${diffusion_sampling}" ] && diffusion_sampling="ddpm"
[ -z "${diffusion_training_loss}" ] && diffusion_training_loss="L1"

[ -z "${num_timesteps}" ] && num_timesteps=5000
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=1e-7
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=2e-3
[ -z "${ddpm_schedule}" ] && ddpm_schedule=sigmoid

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=False
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=False
[ -z "${use_smooth_equviariant_norm}" ] && use_smooth_equviariant_norm=True
[ -z "${num_edges}" ] && num_edges=25600
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=32

[ -z "${psm_validation_mode}" ] && psm_validation_mode=False
[ -z "${use_2d_atom_features}" ] && use_2d_atom_features=True
[ -z "${use_2d_bond_features}" ] && use_2d_bond_features=True
[ -z "${only_use_rotary_embedding_for_protein}" ] && only_use_rotary_embedding_for_protein=True
[ -z "${psm_finetune_mode}" ] && psm_finetune_mode=False
[ -z "${use_hard_dist_loss}" ] && use_hard_dist_loss=True
[ -z "${if_total_energy}" ] && if_total_energy=False
[ -z "${decoder_feat4energy}" ] && decoder_feat4energy=False
[ -z "${disable_data_aug}" ] && disable_data_aug=False
[ -z "${align_x0_in_diffusion_loss}" ] && align_x0_in_diffusion_loss=False

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
# export OMPI_COMM_WORLD_SIZE=1
# export n_gpu=1
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

echo "strategy !!!! ${strategy}"

export wandb=True
[ -z "${ifresume}" ] && ifresume=False

        #   backbone_config.num_layers=9 \
        #   backbone_config.irreps_node_embedding="512x0e+512x1e" \
        #   backbone_config.irreps_head="16x0e+16x1e" \
        #   backbone_config.num_attn_heads=32 \
#
 #3mod-e2dit102-1024
torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
          --config-name=config_psm.yaml \
          wandb=$wandb wandb_group=test wandb_team=$wandb_team wandb_project=$wandb_project \
          wandb_run_name="dit" \
          clean_sample_ratio=$clean_sample_ratio \
          node_type_edge_method=EXCHANGABLE \
          backbone=dit \
          backbone_config=e2former \
          backbone_config.num_layers=4 \
          backbone_config.irreps_node_embedding="1024x0e+1024x1e" \
          backbone_config.irreps_head="32x0e+32x1e" \
          backbone_config.num_attn_heads=32 \
          backbone_config.number_of_basis=32 \
          backbone_config.pbc_max_radius=10 \
          backbone_config.max_radius=30 \
          backbone_config.attn_type=None \
          backbone_config.tp_type=None \
          backbone_config.edge_embedtype='highorder' \
          backbone_config.basis_type='gaussiansmear' \
          backbone_config.attn_biastype='share' \
          backbone_config.add_rope=True \
          encoder_embed_dim=1024 \
          encoder_attention_heads=$num_head \
          encoder_layers=12 \
          num_pred_attn_layer=4 \
          encoder_ffn_embed_dim=$ffn_size \
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
          seed=6666 \
          loadcheck_path=$loadcheck_path \
          ifresume=$ifresume \
          mask_ratio=$mask_ratio \
          d_tilde=$d_tilde \
          strategy=$strategy \
          max_lr=$max_lr \
          diffusion_mode=\"$diffusion_mode\" \
          mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
          complex_mode_prob=\"$complex_mode_prob\" \
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
          rescale_loss_with_std=$rescale_loss_with_std align_x0_in_diffusion_loss=$align_x0_in_diffusion_loss \
          molecule_ref_energy_source=$molecule_ref_energy_source


# torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
#           --config-name=config_psm.yaml \
#           backbone_config=equiformerv2 \
#           backbone=equiformerv2 \
#           backbone_config.embedding_dim=512 \
#           backbone_config.num_heads=32 \
#           backbone_config.order=4 \
#           backbone_config.num_gnn_layers=6 \
#           backbone_config.max_radius=10 \
#           encoder_embed_dim=512 \
#           encoder_attention_heads=$num_head \
#           encoder_layers=$layers \
#           num_pred_attn_layer=$num_pred_attn_layer \
#           encoder_ffn_embed_dim=$ffn_size \
#           droppath_prob=$droppath_prob \
#           attn_dropout=$attn_dropout \
#           act_dropout=$act_dropout \
#           dropout=$dropout \
#           weight_decay=$weight_decay \
#           sandwich_ln=True \
#           data_path=$data_path \
#           data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
#           dataset_split_raito=\"$dataset_split_raito\" \
#           save_dir=$save_dir \
#           seed=6666 \
#           loadcheck_path=$loadcheck_path \
#           ifresume=$ifresume \
#           mask_ratio=$mask_ratio \
#           d_tilde=$d_tilde \
#           strategy=$strategy \
#           max_lr=$max_lr \
#           diffusion_mode=\"$diffusion_mode\" \
#           mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
#           total_num_steps=$total_num_steps \
#           warmup_num_steps=$warmup_num_steps \
#           train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
#           gradient_accumulation_steps=$gradient_accumulation_steps \
#           save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
#           save_batch_interval=$save_batch_interval log_interval=$log_interval \
#           equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
#           pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
#           pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
#           diffusion_noise_std=$diffusion_noise_std fp16=$fp16 \
#           psm_validation_mode=$psm_validation_mode \
#           diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
#           num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
#           ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
#           dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
#           equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler
#           use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
#           use_dali_pipeline=$use_dali_pipeline molecule_energy_loss_ratio=$molecule_energy_loss_ratio \
#           material_energy_loss_ratio=$material_energy_loss_ratio material_force_loss_ratio=$material_force_loss_ratio \
#           preprocess_2d_bond_features_with_cuda=True \
#           AutoGradForce=$AutoGradForce force_head_type=$force_head_type \
#           only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
#           diffusion_training_loss=$diffusion_training_loss use_hard_dist_loss=$use_hard_dist_loss \
#           mm_tensorcore=$mm_tensorcore compile=$compile \
#           if_total_energy=$if_total_energy decoder_feat4energy=$decoder_feat4energy


# torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
#           --config-name=config_psm.yaml \
#           backbone_config=equiformer \
#           backbone=equiformer \
#           backbone_config.irreps_node_embedding="128x0e+128x1e+128x2e+128x3e+128x4e" \
#           backbone_config.num_heads=8 \
#           backbone_config.irreps_head="16x0e+16x1o+16x2e" \
#           backbone_config.num_layers=6 \
#           backbone_config.max_radius=10 \
#           backbone_config.num_heads=$num_head \
#           encoder_embed_dim=512 \
#           encoder_attention_heads=$num_head \
#           encoder_layers=$layers \
#           num_pred_attn_layer=$num_pred_attn_layer \
#           encoder_ffn_embed_dim=$ffn_size \
#           droppath_prob=$droppath_prob \
#           attn_dropout=$attn_dropout \
#           act_dropout=$act_dropout \
#           dropout=$dropout \
#           weight_decay=$weight_decay \
#           sandwich_ln=True \
#           data_path=$data_path \
#           data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
#           dataset_split_raito=\"$dataset_split_raito\" \
#           save_dir=$save_dir \
#           seed=6666 \
#           loadcheck_path=$loadcheck_path \
#           ifresume=$ifresume \
#           mask_ratio=$mask_ratio \
#           d_tilde=$d_tilde \
#           strategy=$strategy \
#           max_lr=$max_lr \
#           diffusion_mode=\"$diffusion_mode\" \
#           mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
#           total_num_steps=$total_num_steps \
#           warmup_num_steps=$warmup_num_steps \
#           train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
#           gradient_accumulation_steps=$gradient_accumulation_steps \
#           save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
#           save_batch_interval=$save_batch_interval log_interval=$log_interval \
#           equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
#           pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
#           pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
#           diffusion_noise_std=$diffusion_noise_std fp16=$fp16 \
#           psm_validation_mode=$psm_validation_mode \
#           diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
#           num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
#           ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
#           dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
#           equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler
#           use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
#           use_dali_pipeline=$use_dali_pipeline molecule_energy_loss_ratio=$molecule_energy_loss_ratio \
#           material_energy_loss_ratio=$material_energy_loss_ratio material_force_loss_ratio=$material_force_loss_ratio \
#           preprocess_2d_bond_features_with_cuda=True \
#           AutoGradForce=$AutoGradForce force_head_type=$force_head_type \
#           only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
#           diffusion_training_loss=$diffusion_training_loss use_hard_dist_loss=$use_hard_dist_loss \
#           mm_tensorcore=$mm_tensorcore compile=$compile \
#           if_total_energy=$if_total_energy decoder_feat4energy=$decoder_feat4energy




# torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
#           --config-name=config_psm.yaml \
#           clean_sample_ratio=1 \
#           wandb=$wandb wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
#           wandb_run_name="graphormer6L_geo4L" \
#           backbone_config=graphormer \
#           backbone=vanillatransformer \
#           encoder_embed_dim=512 \
#           encoder_attention_heads=$num_head \
#           encoder_layers=$layers \
#           num_pred_attn_layer=$num_pred_attn_layer \
#           encoder_ffn_embed_dim=$ffn_size \
#           droppath_prob=$droppath_prob \
#           attn_dropout=$attn_dropout \
#           act_dropout=$act_dropout \
#           dropout=$dropout \
#           weight_decay=$weight_decay \
#           sandwich_ln=True \
#           data_path=$data_path \
#           data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
#           dataset_split_raito=\"$dataset_split_raito\" \
#           save_dir=$save_dir \
#           seed=6666 \
#           loadcheck_path=$loadcheck_path \
#           ifresume=$ifresume \
#           mask_ratio=$mask_ratio \
#           d_tilde=$d_tilde \
#           strategy=$strategy \
#           max_lr=$max_lr \
#           diffusion_mode=\"$diffusion_mode\" \
#           mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
#           total_num_steps=$total_num_steps \
#           warmup_num_steps=$warmup_num_steps \
#           train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
#           gradient_accumulation_steps=$gradient_accumulation_steps \
#           save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
#           save_batch_interval=$save_batch_interval log_interval=$log_interval \
#           equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
#           pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
#           pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
#           diffusion_noise_std=$diffusion_noise_std fp16=$fp16 \
#           psm_validation_mode=$psm_validation_mode \
#           diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
#           num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
#           ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
#           dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
#           equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler
#           use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
#           use_dali_pipeline=$use_dali_pipeline molecule_energy_loss_ratio=$molecule_energy_loss_ratio \
#           material_energy_loss_ratio=$material_energy_loss_ratio material_force_loss_ratio=$material_force_loss_ratio \
#           preprocess_2d_bond_features_with_cuda=True \
#           AutoGradForce=$AutoGradForce force_head_type=$force_head_type \
#           only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
#           diffusion_training_loss=$diffusion_training_loss use_hard_dist_loss=$use_hard_dist_loss \
#           mm_tensorcore=$mm_tensorcore compile=$compile \
#           if_total_energy=$if_total_energy decoder_feat4energy=$decoder_feat4energy



# torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
#           --config-name=config_psm.yaml \
#           backbone_config=graphormer \
#           backbone=graphormer \
#           encoder_embed_dim=512 \
#           encoder_attention_heads=$num_head \
#           encoder_layers=$layers \
#           num_pred_attn_layer=$num_pred_attn_layer \
#           encoder_ffn_embed_dim=$ffn_size \
#           droppath_prob=$droppath_prob \
#           attn_dropout=$attn_dropout \
#           act_dropout=$act_dropout \
#           dropout=$dropout \
#           weight_decay=$weight_decay \
#           sandwich_ln=True \
#           data_path=$data_path \
#           data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
#           dataset_split_raito=\"$dataset_split_raito\" \
#           save_dir=$save_dir \
#           seed=6666 \
#           loadcheck_path=$loadcheck_path \
#           ifresume=$ifresume \
#           mask_ratio=$mask_ratio \
#           d_tilde=$d_tilde \
#           strategy=$strategy \
#           max_lr=$max_lr \
#           diffusion_mode=\"$diffusion_mode\" \
#           mode_prob=\"$mode_prob\" noise_mode=$noise_mode \
#           total_num_steps=$total_num_steps \
#           warmup_num_steps=$warmup_num_steps \
#           train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
#           gradient_accumulation_steps=$gradient_accumulation_steps \
#           save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
#           save_batch_interval=$save_batch_interval log_interval=$log_interval \
#           equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
#           pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
#           pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
#           diffusion_noise_std=$diffusion_noise_std fp16=$fp16 \
#           psm_validation_mode=$psm_validation_mode \
#           diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
#           num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
#           ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
#           dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
#           equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler
#           use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
#           use_dali_pipeline=$use_dali_pipeline molecule_energy_loss_ratio=$molecule_energy_loss_ratio \
#           material_energy_loss_ratio=$material_energy_loss_ratio material_force_loss_ratio=$material_force_loss_ratio \
#           preprocess_2d_bond_features_with_cuda=True \
#           AutoGradForce=$AutoGradForce force_head_type=$force_head_type \
#           only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
#           diffusion_training_loss=$diffusion_training_loss use_hard_dist_loss=$use_hard_dist_loss \
#           mm_tensorcore=$mm_tensorcore compile=$compile \
#           if_total_energy=$if_total_energy decoder_feat4energy=$decoder_feat4energy
