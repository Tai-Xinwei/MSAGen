#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
set_before=$( set -o posix; set | sed -e '/^_=*/d' )

ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${backbone}" ] && backbone=exp3

[ -z "${seed}" ] && seed=666

# [ -z "${layers}" ] && layers=26
# [ -z "${hidden_size}" ] && hidden_size=1536
# [ -z "${ffn_size}" ] && ffn_size=6144
# [ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=8
# [ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=1536
# [ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=1536
# [ -z "${num_head}" ] && num_head=32

[ -z "${layers}" ] && layers=32
[ -z "${hidden_size}" ] && hidden_size=2048
[ -z "${ffn_size}" ] && ffn_size=8192
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=16
[ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=2048
[ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=8192
[ -z "${num_head}" ] && num_head=32

[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=2048
[ -z "${max_residue_num}" ] && max_residue_num=2048
[ -z "${ligand_crop_size}" ] && ligand_crop_size=20.0
[ -z "${max_tokens}" ] && max_tokens=2000
[ -z "${plddt_threshold}" ] && plddt_threshold=60.0

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
[ -z "${max_lr}" ] && max_lr=1e-5
[ -z "${epochs}" ] && epochs=100
[ -z "${total_num_steps}" ] && total_num_steps=2000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=200

[ -z "${train_batch_size}" ] && train_batch_size=16
[ -z "${val_batch_size}" ] && val_batch_size=16
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=1
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=2000000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${val_batch_interval}" ] && val_batch_interval=10000
[ -z "${mode_prob}" ] && mode_prob='0.0,1.0,0.0' #'0.2,0.7,0.1'
[ -z "${complex_mode_prob}" ] && complex_mode_prob='1.0,0.0,0.0,0.0'

[ -z "${data_path}" ] && data_path='/data/peiran/'
# [ -z "${data_path}" ] && data_path='/data/peiran/blob/hai1data/sfm/psm'
[ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,matter-sim-15M,AFDB50-plddt70.lmdb'
[ -z "${dataset_name_list}" ] && dataset_name_list='pm6,mattersim,afdb'
[ -z "${dataset_split_raito}" ] && dataset_split_raito='0.4,0.2,0.4'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="8,4,1"

[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=False
[ -z "${group_optimizer}" ] && group_optimizer=True
[ -z "${group_lr_ratio}" ] && group_lr_ratio=1.0
[ -z "${AutoGradForce}" ] && AutoGradForce=False
[ -z "${NoisePredForce}" ] && NoisePredForce=False
[ -z "${force_head_type}" ] && force_head_type=MLP
[ -z "${force_loss_type}" ] && force_loss_type=L1
[ -z "${molecule_energy_loss_ratio}" ] && molecule_energy_loss_ratio=0.2
[ -z "${molecule_force_loss_ratio}" ] && molecule_force_loss_ratio=2.0
[ -z "${material_energy_loss_ratio}" ] && material_energy_loss_ratio=0.2
[ -z "${material_force_loss_ratio}" ] && material_force_loss_ratio=2.0
[ -z "${energy_per_atom_label_scale}" ] && energy_per_atom_label_scale=1.0
[ -z "${molecule_ref_energy_source}" ] && molecule_ref_energy_source="PubChemQC-B3LYP-PM6/wb97xd3/1.0.0/train"
[ -z "${molecule_outlier_energy_atoms}" ] && molecule_outlier_energy_atoms=""

[ -z "${rescale_loss_with_std}" ] && rescale_loss_with_std=True
[ -z "${use_dali_pipeline}" ] && use_dali_pipeline=False
[ -z "${fp16}" ] && fp16=True
[ -z "${mm_tensorcore}" ] && mm_tensorcore="tf32"
[ -z "${compile}" ] && compile=False

[ -z "${loadcheck_path}" ] && loadcheck_path='/data/peiran/blob/sfmarca100/sfm/sfmexpresults/peiran/psmv1_mi300_edm_exp3_v22_3b_ps_stage1_4/checkpoints/global_step65000/mp_rank_00_model_states.pt'


[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="RELATIVE_POS_VEC_BIAS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=100.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=4096
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=7.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=True
[ -z "${use_fixed_init_lattice_size}" ] && use_fixed_init_lattice_size=True

[ -z "${diffusion_noise_std}" ] && diffusion_noise_std=1.0
[ -z "${diffusion_rescale_coeff}" ] && diffusion_rescale_coeff=1.0
[ -z "${diffusion_mode}" ] && diffusion_mode=edm #epsilon, edm, protea
[ -z "${diffusion_sampling}" ] && diffusion_sampling="ddpm"
[ -z "${diffusion_training_loss}" ] && diffusion_training_loss="L2"
[ -z "${diff_init_lattice_size}" ] && diff_init_lattice_size=10.0

[ -z "${num_timesteps}" ] && num_timesteps=5000
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=1e-7
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=6e-3
[ -z "${ddpm_schedule}" ] && ddpm_schedule=sigmoid #sigmoid, cosine, linaer, quadratic, sqrt

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=False
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=False
[ -z "${use_smooth_equviariant_norm}" ] && use_smooth_equviariant_norm=True
[ -z "${num_edges}" ] && num_edges=25600
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=32

[ -z "${psm_validation_mode}" ] && psm_validation_mode=False
[ -z "${use_2d_atom_features}" ] && use_2d_atom_features=True
[ -z "${use_2d_bond_features}" ] && use_2d_bond_features=False
[ -z "${only_use_rotary_embedding_for_protein}" ] && only_use_rotary_embedding_for_protein=True
[ -z "${psm_finetune_mode}" ] && psm_finetune_mode=True

[ -z "${all_atom}" ] && all_atom=False
[ -z "${use_hard_dist_loss}" ] && use_hard_dist_loss=False
[ -z "${if_total_energy}" ] && if_total_energy=False
[ -z "${decoder_feat4energy}" ] && decoder_feat4energy=True
[ -z "${encoderfeat4noise}" ] && encoderfeat4noise=False
[ -z "${disable_data_aug}" ] && disable_data_aug=False
[ -z "${use_memory_efficient_attention}" ] && use_memory_efficient_attention=False
[ -z "${align_x0_in_diffusion_loss}" ] && align_x0_in_diffusion_loss=True
[ -z "${unified_data_num_workers}" ] && unified_data_num_workers=1

[ -z "${sample_in_validation}" ] && sample_in_validation=False
[ -z "${num_sampling_time}" ] && num_sampling_time=1
[ -z "${sampled_structure_output_path}" ] && sampled_structure_output_path="sample_save_dir"
[ -z "${psm_sample_structure_in_finetune}" ] && psm_sample_structure_in_finetune=False # no middle use
[ -z "${psm_finetune_reset_head}" ] && psm_finetune_reset_head=True

[ -z "${data_basepath}" ] && data_basepath='/fastdata/peiran/psm/bfm_benchmark'
[ -z "${task_name}" ] && task_name='GeneOntology_bp' # EnzymeCommission, GeneOntology_mf, GeneOntology_bp, GeneOntology_cc, stability
[ -z "${save_dir}" ] && save_dir="/data/peiran/output/exp3_3b_prot_${task_name}" #_${train_batch_size}_lr${max_lr}"

[ -z "${wandb_group}" ] && wandb_group="psm_finetune_${task_name}"
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=psm_protein_finetune
[ -z "${wandb_key}" ] && wandb_key=local-138548ae9c9a3b39646af8ae2c4c6d4e22c51385

[ -z "${early_stopping}" ] && early_stopping=True
[ -z "${early_stopping_patience}" ] && early_stopping_patience=10
[ -z "${early_stopping_metric}" ] && early_stopping_metric="f1_max"
[ -z "${early_stopping_mode}" ] && early_stopping_mode="max"
[ -z "${label_normalize}" ] && label_normalize=False
[ -z "${psm_finetune_noise_mode}" ] && psm_finetune_noise_mode="T"
[ -z "${psm_finetune_valid_noise_mode}" ] && psm_finetune_valid_noise_mode="T"


echo -e "\n\n"
echo "==================================MP==========================================="


[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
# [ -z "${n_gpu}" ] && n_gpu=$(rocm-smi | grep -c '^[0-9]') # new for MI250x

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
set_after=$( set -o posix; unset set_before; set | sed -e '/^_=/d' )
diff  <(echo "$set_before") <(echo "$set_after") | sed -e 's/^> //' -e '/^[[:digit:]].*/d'
# hack from https://stackoverflow.com/questions/1305237/how-to-list-variables-declared-in-script-in-bash

# export NCCL_ASYNC_ERROR_HADNLING=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_DISABLE=1
export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=1

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

torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/finetune_protein_understanding.py \
          --config-name=config_psm.yaml \
          backbone_config=graphormer \
          backbone=$backbone \
          encoder_attention_heads=$num_head \
          encoder_layers=$layers \
          num_pred_attn_layer=$num_pred_attn_layer \
          encoder_ffn_embed_dim=$ffn_size \
          encoder_embed_dim=$hidden_size \
          decoder_hidden_dim=$decoder_hidden_dim \
          decoder_ffn_dim=$decoder_ffn_dim \
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
          seed=7778 \
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
          diffusion_noise_std=$diffusion_noise_std diffusion_rescale_coeff=$diffusion_rescale_coeff \
          fp16=$fp16 use_memory_efficient_attention=$use_memory_efficient_attention \
          psm_validation_mode=$psm_validation_mode num_edges=$num_edges num_3d_bias_kernel=$num_3d_bias_kernel \
          diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
          num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
          ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          clean_sample_ratio=$clean_sample_ratio \
          use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
          use_dali_pipeline=$use_dali_pipeline \
          molecule_energy_loss_ratio=$molecule_energy_loss_ratio molecule_force_loss_ratio=$molecule_force_loss_ratio \
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
          loadcheck_path=$loadcheck_path encoderfeat4noise=$encoderfeat4noise \
          molecule_outlier_energy_atoms=$molecule_outlier_energy_atoms molecule_ref_energy_source=$molecule_ref_energy_source \
          max_residue_num=$max_residue_num ligand_crop_size=$ligand_crop_size plddt_threshold=$plddt_threshold \
          unified_data_num_workers=$unified_data_num_workers group_optimizer=$group_optimizer group_lr_ratio=$group_lr_ratio \
          task_name=$task_name \
          data_basepath=$data_basepath \
          head_dropout=0.1 \
          label_normalize=$label_normalize \
          checkpoint_dir="" \
          which_set="valid" \
          calculate_metrics=True \
          early_stopping=$early_stopping early_stopping_patience=$early_stopping_patience \
          early_stopping_metric=$early_stopping_metric early_stopping_mode=$early_stopping_mode \
          psm_finetune_noise_mode=$psm_finetune_noise_mode psm_finetune_valid_noise_mode=$psm_finetune_valid_noise_mode \


# RESULT=$?
# if [ $RESULT -eq 0 ]; then
#   echo "Training finished successfully"
# else
#   echo "Training finished with error, not continue"
#   exit 1
# fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/test_protein_understanding.py \
          --config-name=config_psm.yaml \
          backbone_config=graphormer \
          backbone=$backbone \
          encoder_attention_heads=$num_head \
          encoder_layers=$layers \
          num_pred_attn_layer=$num_pred_attn_layer \
          encoder_ffn_embed_dim=$ffn_size \
          encoder_embed_dim=$hidden_size \
          decoder_hidden_dim=$decoder_hidden_dim \
          decoder_ffn_dim=$decoder_ffn_dim \
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
          seed=7778 \
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
          diffusion_noise_std=$diffusion_noise_std diffusion_rescale_coeff=$diffusion_rescale_coeff \
          fp16=$fp16 use_memory_efficient_attention=$use_memory_efficient_attention \
          psm_validation_mode=$psm_validation_mode num_edges=$num_edges num_3d_bias_kernel=$num_3d_bias_kernel \
          diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
          num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
          ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          clean_sample_ratio=$clean_sample_ratio \
          use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
          use_dali_pipeline=$use_dali_pipeline \
          molecule_energy_loss_ratio=$molecule_energy_loss_ratio molecule_force_loss_ratio=$molecule_force_loss_ratio \
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
          loadcheck_path=$save_dir/checkpoint_best.pt encoderfeat4noise=$encoderfeat4noise \
          molecule_outlier_energy_atoms=$molecule_outlier_energy_atoms molecule_ref_energy_source=$molecule_ref_energy_source \
          max_residue_num=$max_residue_num ligand_crop_size=$ligand_crop_size plddt_threshold=$plddt_threshold \
          unified_data_num_workers=$unified_data_num_workers group_optimizer=$group_optimizer group_lr_ratio=$group_lr_ratio \
          task_name=$task_name \
          data_basepath=$data_basepath \
          head_dropout=0.1 \
          label_normalize=$label_normalize \
          checkpoint_dir="" \
          which_set="test" \
          calculate_metrics=True \
          early_stopping=$early_stopping early_stopping_patience=$early_stopping_patience \
          early_stopping_metric=$early_stopping_metric early_stopping_mode=$early_stopping_mode \
          psm_finetune_noise_mode=$psm_finetune_noise_mode psm_finetune_valid_noise_mode=$psm_finetune_valid_noise_mode \
