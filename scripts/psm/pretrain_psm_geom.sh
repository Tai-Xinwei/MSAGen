#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
set_before=$( set -o posix; set | sed -e '/^_=*/d' )

ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=1536
[ -z "${ffn_size}" ] && ffn_size=6144
[ -z "${decoder_ffn_dim}" ]  && decoder_ffn_dim=6144
[ -z "${num_head}" ] && num_head=32
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=14
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=1024
[ -z "${max_tokens}" ] && max_tokens=2000
# [ -z "${max_tokens}" ] && max_tokens=36000

[ -z "${dropout}" ] && dropout=0.1
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${noise_mode}" ] && noise_mode=diff

[ -z "${mask_ratio}" ] && mask_ratio=0.5
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=2e-4
[ -z "${total_num_steps}" ] && total_num_steps=2000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=12000
[ -z "${train_batch_size}" ] && train_batch_size=64
[ -z "${val_batch_size}" ] && val_batch_size=64
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=8
[ -z "${strategy}" ] && strategy=DDP
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=2500
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=1000
[ -z "${val_batch_interval}" ] && val_batch_interval=30000

[ -z "${mode_prob}" ] && mode_prob='0.1,0.2,0.7' #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full
# [ -z "${mode_prob}" ] && mode_prob='0.0,0.0,0.0,1.0' # prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full

[ -z "${data_path}" ] && data_path='/home/v-xixianliu/work/xixian/data/psm'
# [ -z "${data_path}" ] && data_path='/data/peiran/blob/hai1data/sfm/psm'
[ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,matter-sim-15M-force-filtered-merged,AFDB70-plddt70.lmdb,matter-sim-15M-merged,ur50_23_bpe_pack1536.lmdb,20240630_PDB_Training_Data,20240630_PDB_Training_Data'
[ -z "${dataset_name_list}" ] && dataset_name_list='pm6,mattersim,afdb,mattersim,ur50,pdb,pdbcomplexmultimer'
[ -z "${dataset_split_raito}" ] && dataset_split_raito='0.2,0.05,0.3,0.15,0.1,0.1,0.1'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="8,4,2,4,2,2,2"
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True

[ -z "${loadcheck_path}" ] && loadcheck_path=''
[ -z "${save_dir}" ] && save_dir='/home/v-xixianliu/work/xixian/checkpoint/test'
# [ -z "${save_dir}" ] && save_dir='/home/peiran/FMproj/output/'
[ -z "${dataset_name}" ] && dataset_name="."
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0

[ -z "${wandb_group}" ] && wandb_group=v-xixianliu
[ -z "${wandb_team}" ] && wandb_team=v-xixianliu
[ -z "${wandb_project}" ] && wandb_project=psm_dev
[ -z "${wandb_key}" ] && wandb_key=local-92e9aa662fb8066a31846fb8e57abd4e90ed09d8

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="AUGMENTED_RELATIVE_POS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=20.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=256
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=5.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=True
[ -z "${diffusion_noise_std}" ] && diffusion_noise_std=10.0

[ -z "${diff_init_lattice_size}" ] && diff_init_lattice_size=10.0
[ -z "${diffusion_sampling}" ] && diffusion_sampling="ddpm"
[ -z "${num_timesteps}" ] && num_timesteps=5000
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=1e-7
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=2e-3
[ -z "${ddpm_schedule}" ] && ddpm_schedule=sigmoid

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=True
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=False

[ -z "${clean_sample_ratio}" ] && clean_sample_ratio=0.5

[ -z "${fp16}" ] && fp16=False

[ -z "${psm_validation_mode}" ] && psm_validation_mode=False
[ -z "${sample_in_validation}" ] && sample_in_validation=False
[ -z "${num_sampling_time}" ] && num_sampling_time=1
[ -z "${sampled_structure_output_path}" ] && sampled_structure_output_path="sample_save_dir"
[ -z "${psm_finetune_mode}" ] && psm_finetune_mode=False
[ -z "${psm_sample_structure_in_finetune}" ] && psm_sample_structure_in_finetune=False
[ -z "${psm_finetune_reset_head}" ] && psm_finetune_reset_head=True
[ -z "${val_batch_log_all_metric}" ] && val_batch_log_all_metric=False
[ -z "${psm_validate_for_train_set}" ] && psm_validate_for_train_set=False

[ -z "${rescale_loss_with_std}" ] && rescale_loss_with_std=True
[ -z "${only_use_rotary_embedding_for_protein}" ] && only_use_rotary_embedding_for_protein=True
[ -z "${use_dali_pipeline}" ] && use_dali_pipeline=False

[ -z "${use_memory_efficient_attention}" ] && use_memory_efficient_attention=True

[ -z "${psm_matbench_task_name}" ] && psm_matbench_task_name=matbench_dielectric
[ -z "${psm_matbench_fold_id}" ] && psm_matbench_fold_id=0
[ -z "${psm_finetune_valid_noise_mode}" ] && psm_finetune_valid_noise_mode="zero"
[ -z "${force_loss_type}" ] && force_loss_type="L1"
[ -z "${diffusion_training_loss}" ] && diffusion_training_loss="L1"
[ -z "${align_x0_in_diffusion_loss}" ] && align_x0_in_diffusion_loss=False

[ -z "${use_smooth_softmax}" ] && use_smooth_softmax=True
[ -z "${smooth_factor}" ] && smooth_factor=20.0

[ -z "${num_edges}" ] && num_edges=25600
[ -z "${no_rotary_embedding_for_vector}" ] && no_rotary_embedding_for_vector=False
[ -z "${node_type_edge_method}" ] && node_type_edge_method=NON_EXCHANGABLE
[ -z "${force_head_type}" ] && force_head_type=GATED_EQUIVARIANT
[ -z "${mlm_from_decoder_feature}" ] && mlm_from_decoder_feature=False
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128
[ -z "${use_smooth_equviariant_norm}" ] && use_smooth_equviariant_norm=True
[ -z "${AutoGradForce}" ] && AutoGradForce=False

[ -z "${unified_data_num_workers}" ] && unified_data_num_workers=0
[ -z "${use_fp32_in_decoder}" ] && use_fp32_in_decoder=True
[ -z "${material_force_loss_ratio}" ] && material_force_loss_ratio=1
[ -z "${material_energy_loss_ratio}" ] && material_energy_loss_ratio=1
[ -z "${molecule_energy_loss_ratio}" ] && molecule_energy_loss_ratio=1
[ -z "${energy_per_atom_label_scale}" ] && energy_per_atom_label_scale=0.05


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

torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_psm.py \
          --config-name=config_psm.yaml \
          backbone_config=graphormer \
          backbone=geomformer \
          encoder_attention_heads=$num_head \
          encoder_layers=$layers \
          encoder_ffn_embed_dim=$ffn_size \
          encoder_embed_dim=$hidden_size \
          droppath_prob=$droppath_prob \
          attn_dropout=$attn_dropout \
          act_dropout=$act_dropout \
          dropout=$dropout \
          weight_decay=$weight_decay \
          sandwich_ln=True \
          add_3d=True \
          data_path=$data_path \
          data_path_list=\"$data_path_list\" dataset_name_list=\"$dataset_name_list\" \
          dataset_split_raito=\"$dataset_split_raito\" \
          save_dir=$save_dir \
          seed=12345 \
          ifresume=True \
          mask_ratio=$mask_ratio \
          noise_scale=$noise_scale \
          num_pred_attn_layer=$num_pred_attn_layer \
          d_tilde=$d_tilde \
          strategy=$strategy \
          max_lr=$max_lr \
          mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
          use_2d_atom_features=True use_2d_bond_features=True \
          total_num_steps=$total_num_steps \
          warmup_num_steps=$warmup_num_steps \
          train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
          save_batch_interval=$save_batch_interval log_interval=$log_interval loadcheck_path=$loadcheck_path \
          equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
          pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
          pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
          diffusion_noise_std=$diffusion_noise_std fp16=$fp16 \
          diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
          num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
          ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          clean_sample_ratio=$clean_sample_ratio \
          sample_in_validation=$sample_in_validation \
          sampled_structure_output_path=$sampled_structure_output_path \
          psm_validation_mode=$psm_validation_mode \
          num_sampling_time=$num_sampling_time \
          psm_finetune_mode=$psm_finetune_mode \
          psm_sample_structure_in_finetune=$psm_sample_structure_in_finetune \
          psm_finetune_reset_head=$psm_finetune_reset_head \
          val_batch_log_all_metric=$val_batch_log_all_metric \
          psm_validate_for_train_set=$psm_validate_for_train_set \
          rescale_loss_with_std=$rescale_loss_with_std \
          only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
          use_memory_efficient_attention=$use_memory_efficient_attention \
          decoder_ffn_dim=$decoder_ffn_dim \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
          use_dali_pipeline=$use_dali_pipeline \
          wandb_run_name=$wandb_run_name val_batch_interval=$val_batch_interval \
          psm_matbench_task_name=$psm_matbench_task_name \
          psm_matbench_fold_id=$psm_matbench_fold_id \
          psm_finetune_valid_noise_mode=$psm_finetune_valid_noise_mode \
          diffusion_training_loss=$diffusion_training_loss \
          force_loss_type=$force_loss_type \
          ++energy_per_atom_label_scale=$energy_per_atom_label_scale ++molecule_energy_per_atom_std_override=1.0 \
          align_x0_in_diffusion_loss=$align_x0_in_diffusion_loss \
          num_edges=$num_edges \
          no_rotary_embedding_for_vector=$no_rotary_embedding_for_vector \
          node_type_edge_method=$node_type_edge_method \
          force_head_type=$force_head_type \
          mlm_from_decoder_feature=$mlm_from_decoder_feature \
          num_3d_bias_kernel=$num_3d_bias_kernel \
          unified_data_num_workers=$unified_data_num_workers \
          use_fp32_in_decoder=$use_fp32_in_decoder \
          material_force_loss_ratio=$material_force_loss_ratio \
          material_energy_loss_ratio=$material_energy_loss_ratio \
          molecule_energy_loss_ratio=$molecule_energy_loss_ratio \
          use_smooth_softmax=$use_smooth_softmax \
          smooth_factor=$smooth_factor \
          use_smooth_equviariant_norm=$use_smooth_equviariant_norm \
          AutoGradForce=$AutoGradForce \
