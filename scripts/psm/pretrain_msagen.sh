#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

[ -z "${backbone}" ] && backbone=exp3

# [ -z "${layers}" ] && layers=32
# [ -z "${hidden_size}" ] && hidden_size=3072
# [ -z "${ffn_size}" ] && ffn_size=12288
# [ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=32
# [ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=3072
# [ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=12288

# [ -z "${layers}" ] && layers=32
# [ -z "${hidden_size}" ] && hidden_size=2048
# [ -z "${ffn_size}" ] && ffn_size=8192
# [ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=16
# [ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=2048
# [ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=8192

# [ -z "${layers}" ] && layers=26
# [ -z "${hidden_size}" ] && hidden_size=1536
# [ -z "${ffn_size}" ] && ffn_size=6144
# [ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=8
# [ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=1536
# [ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=1536

#150M
[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=1024
[ -z "${ffn_size}" ] && ffn_size=4096
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=12
[ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=1024
[ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=1024
#85M
# [ -z "${layers}" ] && layers=12
# [ -z "${hidden_size}" ] && hidden_size=768
# [ -z "${ffn_size}" ] && ffn_size=3072
# [ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=12
# [ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=768
# [ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=768
#38M
# [ -z "${layers}" ] && layers=12
# [ -z "${hidden_size}" ] && hidden_size=512
# [ -z "${ffn_size}" ] && ffn_size=2048
# [ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=12
# [ -z "${decoder_hidden_dim}" ] && decoder_hidden_dim=512
# [ -z "${decoder_ffn_dim}" ] && decoder_ffn_dim=2048

[ -z "${num_head}" ] && num_head=32
[ -z "${atom_loss_coeff}" ] && atom_loss_coeff=1.0
[ -z "${pos_loss_coeff}" ] && pos_loss_coeff=1.0
[ -z "${max_length}" ] && max_length=384
[ -z "${max_residue_num}" ] && max_residue_num=384
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
[ -z "${max_lr}" ] && max_lr=4e-5
[ -z "${total_num_steps}" ] && total_num_steps=2000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=1000
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=32
[ -z "${strategy}" ] && strategy=Zero1
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=2000
[ -z "${log_interval}" ] && log_interval=20
[ -z "${epochs}" ] && epochs=1000
[ -z "${val_batch_interval}" ] && val_batch_interval=10000
[ -z "${mode_prob}" ] && mode_prob='0.0,1.0,0.0' #'0.2,0.7,0.1'
[ -z "${complex_mode_prob}" ] && complex_mode_prob='1.0,0.0,0.0,0.0' #'0.6,0.2,0.1,0.1' #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full
# [ -z "${complex_mode_prob}" ] && complex_mode_prob='0.5,0.0,0.0,0.5' #'0.6,0.2,0.1,0.1' #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full

[ -z "${data_path}" ] && data_path='../msadata'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="32"

# [ -z "${data_path_list}" ] && data_path_list='20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="2"

# [ -z "${data_path_list}" ] && data_path_list='matter-sim-15M-merged'
# [ -z "${dataset_name_list}" ] && dataset_name_list='mattersim'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='8'

# [ -z "${data_path_list}" ] && data_path_list='matter-gen-force-filtered'
# [ -z "${dataset_name_list}" ] && dataset_name_list='mattersim'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='2'

# [ -z "${data_path_list}" ] && data_path_list='matter-gen-force-filtered,matter-sim-15M-merged'
# [ -z "${dataset_name_list}" ] && dataset_name_list='mattersim,mattersim'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.5,0.5'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='8,8'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,AFDB50-plddt70.lmdb,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3,afdb,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.5,0.25,0.25'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='80,12,6'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,matter-gen-force-filtered,matter-sim-15M-merged,AFDB50-plddt70.lmdb,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3,mattersim,mattersim,afdb,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.2,0.05,0.15,0.4,0.2'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='16,2,2,4,1'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='32,8,8,8,2'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='128,32,32,32,8'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,AFDB50-plddt70.lmdb,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3,afdb,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.4,0.4,0.2'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='16,2,2,4,1'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='32,8,8,8,2'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='32,32,8'

# [ -z "${data_path_list}" ] && data_path_list='AFDB50-plddt70.lmdb,20240630_PDB_Training_Data,20240101_PDB_Training_Data,MGnify'
# [ -z "${dataset_name_list}" ] && dataset_name_list='afdb,pdbcomplexmultimer,pdb,mgnify'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.6,0.20,0.05,0.15'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='32,8,32,32'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='4,1,4,4'

# [ -z "${data_path_list}" ] && data_path_list='AFDB50-plddt70.lmdb,20240101_PDB_Training_Data,MGnify'
# [ -z "${dataset_name_list}" ] && dataset_name_list='afdb,pdb,mgnify'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.60,0.10,0.30'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='4,4,4'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,AFDB50-plddt70.lmdb,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3,afdb,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.4,0.4,0.2'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='80,12,6'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,matter-sim-15M-force-filtered-merged,AFDB50-plddt70.lmdb,matter-sim-15M-merged,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3,mattersim,afdb,mattersim,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.3,0.05,0.4,0.15,0.1'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='32,8,8,8'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='16,2,2,2,1'

# [ -z "${data_path_list}" ] && data_path_list='PubChemQC-B3LYP-PM6,AFDB50-plddt70.lmdb,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pm6-wb97xd3,afdb,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.5,0.4,0.1'
# # [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='32,8,8,8'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='16,4,1'

# [ -z "${data_path_list}" ] && data_path_list='20240630_PDB_Training_Data,20240101_PDB_Training_Data,AFDB50-plddt70.lmdb'
# [ -z "${dataset_name_list}" ] && dataset_name_list='pdbcomplexmultimer,pdb,afdb'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.2,0.05,0.75'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='2,8,8'

# [ -z "${data_path_list}" ] && data_path_list='UniProt90-UniRef50-updated-plddt70-reduce.lmdb'
# [ -z "${dataset_name_list}" ] && dataset_name_list='esm'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="16"

# [ -z "${data_path_list}" ] && data_path_list='AFDB50-plddt70.lmdb'
# [ -z "${dataset_name_list}" ] && dataset_name_list='afdb'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="4"

[ -z "${data_path_list}" ] && data_path_list='protein_msa_40_0.1_3k.lmdb'
[ -z "${dataset_name_list}" ] && dataset_name_list='msageneration'
[ -z "${dataset_split_raito}" ] && dataset_split_raito='1.0'
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="1"

# [ -z "${data_path_list}" ] && data_path_list='AFDB50-plddt70.lmdb,AFDB90-plddt60to70-reduce.lmdb,MGnify,20240630_PDB_Training_Data,PubChemQC-B3LYP-PM6'
# [ -z "${dataset_name_list}" ] && dataset_name_list='afdb,esm,mgnify,pdbcomplexmultimer,pm6-wb97xd3'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.3,0.1,0.2,0.2,0.2'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='4,4,4,4,32'

# [ -z "${data_path_list}" ] && data_path_list='AFDB50-plddt70.lmdb,MGnify,20240630_PDB_Training_Data'
# [ -z "${dataset_name_list}" ] && dataset_name_list='afdb,mgnify,pdbcomplexmultimer'
# [ -z "${dataset_split_raito}" ] && dataset_split_raito='0.5,0.3,0.2'
# [ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size='4,4,4'

[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${group_optimizer}" ] && group_optimizer=False
[ -z "${group_lr_ratio}" ] && group_lr_ratio=1.0
[ -z "${AutoGradForce}" ] && AutoGradForce=False
[ -z "${NoisePredForce}" ] && NoisePredForce=False
[ -z "${force_head_type}" ] && force_head_type=MLP
[ -z "${force_loss_type}" ] && force_loss_type=L1
[ -z "${molecule_energy_loss_ratio}" ] && molecule_energy_loss_ratio=0.1
[ -z "${molecule_force_loss_ratio}" ] && molecule_force_loss_ratio=0.5
[ -z "${material_energy_loss_ratio}" ] && material_energy_loss_ratio=0.1
[ -z "${material_force_loss_ratio}" ] && material_force_loss_ratio=0.5
[ -z "${energy_per_atom_label_scale}" ] && energy_per_atom_label_scale=1.0
[ -z "${molecule_ref_energy_source}" ] && molecule_ref_energy_source="PubChemQC-B3LYP-PM6/wb97xd3/1.0.0/train"
[ -z "${molecule_outlier_energy_atoms}" ] && molecule_outlier_energy_atoms=""

[ -z "${rescale_loss_with_std}" ] && rescale_loss_with_std=True
[ -z "${use_dali_pipeline}" ] && use_dali_pipeline=False
[ -z "${fp16}" ] && fp16=False
[ -z "${bf16}" ] && bf16=False
[ -z "${mm_tensorcore}" ] && mm_tensorcore="tf32"
[ -z "${compile}" ] && compile=False

[ -z "${loadcheck_path}" ] && loadcheck_path=''


[ -z "${wandb_run_name}" ] && wandb_run_name=MSAGen-$(date +"%Y%m%d")-lr${max_lr}-bsz1_${gradient_accumulation_steps}-steps${total_num_steps}-warm${warmup_num_steps}_local
[ -z "${wandb_group}" ] && wandb_group=msagen
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=MSAGen
[ -z "${wandb_key}" ] && wandb_key=local-4475b85516f93bca7c53acde577024463126c48c

[ -z "${save_dir}" ] && save_dir=/psm/sfmexpresults/xinwei/MSAGen/${wandb_run_name}

random_number=$((RANDOM))
echo "Random number: ${random_number}"
[ -z "${seed}" ] && seed=$random_number

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="RELATIVE_POS_VEC_BIAS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=40.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=512
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=7.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=True

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


[ -z "${use_bond_loss}" ] && use_bond_loss=False
[ -z "${psm_validation_mode}" ] && psm_validation_mode=False
[ -z "${sample_in_validation}" ] && sample_in_validation=False
[ -z "${use_2d_atom_features}" ] && use_2d_atom_features=True
[ -z "${use_2d_bond_features}" ] && use_2d_bond_features=False
[ -z "${only_use_rotary_embedding_for_protein}" ] && only_use_rotary_embedding_for_protein=True
[ -z "${psm_finetune_mode}" ] && psm_finetune_mode=False
[ -z "${use_hard_dist_loss}" ] && use_hard_dist_loss=False
[ -z "${if_total_energy}" ] && if_total_energy=False
[ -z "${decoder_feat4energy}" ] && decoder_feat4energy=False
[ -z "${encoderfeat4noise}" ] && encoderfeat4noise=False
[ -z "${encoderfeat4mlm}" ] && encoderfeat4mlm=True
[ -z "${disable_data_aug}" ] && disable_data_aug=False
[ -z "${use_memory_efficient_attention}" ] && use_memory_efficient_attention=False
[ -z "${align_x0_in_diffusion_loss}" ] && align_x0_in_diffusion_loss=True
[ -z "${unified_data_num_workers}" ] && unified_data_num_workers=1


echo -e "\n\n"
echo "==================================MP==========================================="
# [ -z "${n_gpu}" ] && n_gpu=$(rocm-smi | grep -c '^[0-9]') # new for MI250x
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
                      --master_port 11111"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
  fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"
export OMP_NUM_THREADS=16

# cp sfm/utils/barrier.py . && touch READY && python barrier_amd.py $OMPI_COMM_WORLD_SIZE $OMPI_COMM_WORLD_RANK

DDP_TIMEOUT_MINUTES=3000 torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/pretrain_msagen.py \
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
          seed=$seed \
          mask_ratio=$mask_ratio \
          d_tilde=$d_tilde \
          strategy=$strategy \
          max_lr=$max_lr \
          diffusion_mode=\"$diffusion_mode\" \
          mode_prob=\"$mode_prob\" noise_mode=$noise_mode\
          complex_mode_prob=\"$complex_mode_prob\" \
          total_num_steps=$total_num_steps \
          warmup_num_steps=$warmup_num_steps \
          sample_in_validation=$sample_in_validation \
          train_batch_size=$train_batch_size val_batch_size=$val_batch_size max_length=$max_length \
          gradient_accumulation_steps=$gradient_accumulation_steps \
          save_epoch_interval=$save_epoch_interval total_num_epochs=$epochs \
          save_batch_interval=$save_batch_interval log_interval=$log_interval \
          equivar_vec_init=$equivar_vec_init pbc_use_local_attention=$pbc_use_local_attention \
          pbc_cutoff=$pbc_cutoff pbc_expanded_num_cell_per_direction=$pbc_expanded_num_cell_per_direction \
          pbc_expanded_token_cutoff=$pbc_expanded_token_cutoff pbc_multigraph_cutoff=$pbc_multigraph_cutoff \
          diffusion_noise_std=$diffusion_noise_std diffusion_rescale_coeff=$diffusion_rescale_coeff \
          fp16=$fp16 bf16=$bf16 use_memory_efficient_attention=$use_memory_efficient_attention \
          psm_validation_mode=$psm_validation_mode num_edges=$num_edges num_3d_bias_kernel=$num_3d_bias_kernel \
          diff_init_lattice_size=$diff_init_lattice_size diffusion_sampling=$diffusion_sampling \
          num_timesteps=$num_timesteps ddpm_beta_start=$ddpm_beta_start \
          ddpm_beta_end=$ddpm_beta_end ddpm_schedule=$ddpm_schedule \
          dataset_micro_batch_size=\"$dataset_micro_batch_size\" equivar_use_linear_bias=$equivar_use_linear_bias \
          equivar_use_attention_bias=$equivar_use_attention_bias use_unified_batch_sampler=$use_unified_batch_sampler \
          clean_sample_ratio=$clean_sample_ratio \
          use_2d_atom_features=$use_2d_atom_features use_2d_bond_features=$use_2d_bond_features \
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project wandb_run_name=$wandb_run_name \
          use_dali_pipeline=$use_dali_pipeline \
          molecule_energy_loss_ratio=$molecule_energy_loss_ratio molecule_force_loss_ratio=$molecule_force_loss_ratio \
          material_energy_loss_ratio=$material_energy_loss_ratio material_force_loss_ratio=$material_force_loss_ratio \
          energy_per_atom_label_scale=$energy_per_atom_label_scale molecule_energy_per_atom_std_override=1.0 \
          preprocess_2d_bond_features_with_cuda=True use_smooth_equviariant_norm=$use_smooth_equviariant_norm \
          AutoGradForce=$AutoGradForce force_head_type=$force_head_type psm_finetune_mode=$psm_finetune_mode \
          only_use_rotary_embedding_for_protein=$only_use_rotary_embedding_for_protein \
          diffusion_training_loss=$diffusion_training_loss use_hard_dist_loss=$use_hard_dist_loss \
          mm_tensorcore=$mm_tensorcore compile=$compile disable_data_aug=$disable_data_aug \
          if_total_energy=$if_total_energy decoder_feat4energy=$decoder_feat4energy use_bond_loss=$use_bond_loss \
          NoisePredForce=$NoisePredForce force_loss_type=$force_loss_type \
          rescale_loss_with_std=$rescale_loss_with_std align_x0_in_diffusion_loss=$align_x0_in_diffusion_loss \
          loadcheck_path=$loadcheck_path encoderfeat4noise=$encoderfeat4noise encoderfeat4mlm=$encoderfeat4mlm \
          molecule_outlier_energy_atoms=$molecule_outlier_energy_atoms molecule_ref_energy_source=$molecule_ref_energy_source \
          max_residue_num=$max_residue_num ligand_crop_size=$ligand_crop_size plddt_threshold=$plddt_threshold \
          unified_data_num_workers=$unified_data_num_workers group_optimizer=$group_optimizer group_lr_ratio=$group_lr_ratio \
          # ifresume=True \

sleep infinity
