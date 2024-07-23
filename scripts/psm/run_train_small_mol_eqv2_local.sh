export HYDRA_FULL_ERROR=1
export load_ckpt=False
#   mkdir /blob/pfmexp/output
#   mkdir /blob/experiment/psm/psmV0test_0507
#   mkdir /blob/experiment/psm/psmV0test_0507/checkpoints

export pbc_cutoff=20.0
export pbc_expanded_num_cell_per_direction=5
export pbc_expanded_token_cutoff=256
export pbc_multigraph_cutoff=5.0
export pbc_use_local_attention=False

export num_pred_attn_layer=4

export dataset_split_raito=0.4,0.2,0.4

export save_batch_interval=2500
export train_batch_size=16
export val_batch_size=16
export gradient_accumulation_steps=4
export val_batch_interval=0


export total_num_steps=2000000
export warmup_num_steps=12000
export max_lr=1.5e-4

export diffusion_noise_std=0
export equivar_vec_init=ZERO_CENTERED_POS
export strategy=DDP
export fp16=False
export clean_sample_ratio=1.0

export diff_init_lattice_size=10.0
export diffusion_sampling="ddpm"
export num_timesteps=5000
export ddpm_beta_start=1e-7
export ddpm_beta_end=2e-3
export ddpm_schedule=sigmoid

export equivar_use_linear_bias=True
export equivar_use_attention_bias=True

export dataset_micro_batch_size="16"
export use_unified_batch_sampler=True
export rescale_loss_with_std=True
export only_use_rotary_embedding_for_protein=True
export use_memory_efficient_attention=False

# Jia
export save_epoch_interval=20
export wandb_team=ai4s-sfm
export wandb_group=psm_finetune_md22
export wandb_project=psm_dev
export WANDB_API_KEY="local-4e57d49a2495e8001e871d18542a19210e851a6f" #Jia's key
export wandb_key="${WANDB_API_KEY}" #Jia's key
export wandb_run_name='equiformer_v2_md22_random-0626-kcal-3'

export backbone='equiformerv2'
export backbone_config='equiformerv2'
export psm_finetune_mode=true

export data_path='/home/zhangjia/working/'
export data_path_list='sfm_data/MD22/AT_AT_CG_CG/radius3_broadcast'
export dataset_name_list='AT_AT_CG_CG'
export dataset_split_raito='1.0'
export loadcheck_path='../sfm_data/pubchem-pm6-diffusion-molecule-protein-periodic-16xG8-fp32-ddp-unified-sampler-continued-fastpreprocess/checkpoint_E1_B66933.pt'
export save_dir="../sfm_data/checkpoints/$wandb_run_name"
python setup_cython.py build_ext --inplace
bash ./scripts/psm/finetune_psm_equiformerv2.sh
# bash ./scripts/psm/pretrain_psm.sh
