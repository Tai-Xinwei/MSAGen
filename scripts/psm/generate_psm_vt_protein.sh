#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
ulimit -c unlimited


if [ $# == 0 ]; then
  tmpdir=$(mktemp -d)
  #tmpdir="/home/peiranjin/expresult/psmexp/output/psmv1_vt_v3/"
  fasta_list="$tmpdir/fasta_list"
  output_dir="$tmpdir"
  echo ">7vty_A length=90" > "$tmpdir/7vty_A.fasta"
  echo "AKARDKLEENRDLIVERLKVDEIADFMIEKGELTEEEKKKVDAEDSERKRAEKLVEIVMKMDDAAVKAFYDALKAKGYSDLASLLESGLC" >> "$tmpdir/7vty_A.fasta"
  echo "$tmpdir/7vty_A.fasta" > "$fasta_list"
elif [ $# == 2 ]; then
  fasta_list=$1
  output_dir=$2
else
  echo "Default: bash $0"
  echo "Usage: bash $0 <fasta_list> <output_dir>"
  exit 1
fi


export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER="GNU"

[ -z "${layers}" ] && layers=24
[ -z "${hidden_size}" ] && hidden_size=1024
[ -z "${ffn_size}" ] && ffn_size=4096
[ -z "${num_head}" ] && num_head=32
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
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
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${noise_mode}" ] && noise_mode=diff

[ -z "${mask_ratio}" ] && mask_ratio=0.0
[ -z "${clean_sample_ratio}" ] && clean_sample_ratio=0.0

[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=1.5e-4
[ -z "${total_num_steps}" ] && total_num_steps=2000000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=12000
[ -z "${train_batch_size}" ] && train_batch_size=1024
[ -z "${val_batch_size}" ] && val_batch_size=1024
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${strategy}" ] && strategy=DDP
[ -z "${save_epoch_interval}" ] && save_epoch_interval=1
[ -z "${save_batch_interval}" ] && save_batch_interval=2500
[ -z "${log_interval}" ] && log_interval=100
[ -z "${epochs}" ] && epochs=1000
[ -z "${val_batch_interval}" ] && val_batch_interval=0

[ -z "${mode_prob}" ] && mode_prob="0.1,0.2,0.6,0.1" #sss prob of independent mask_pos==mask_type, mask_pos==full, mask_type==full

#[ -z "${data_path}" ] && data_path="/data/peiran/blob/hai1data/sfm/psm"
[ -z "${data_path}" ] && data_path="/casp/jianwzhu/workspace/SFM_Evaluation/run_sfm/sfmblob/psm"
[ -z "${data_path_list}" ] && data_path_list="PubChemQC-B3LYP-PM6,matter-sim-15M,AFDB50-plddt70.lmdb"
[ -z "${dataset_name_list}" ] && dataset_name_list="pm6,mattersim,afdb"
[ -z "${dataset_split_raito}" ] && dataset_split_raito="0.4,0.2,0.4"
[ -z "${dataset_micro_batch_size}" ] && dataset_micro_batch_size="16,4,2"
[ -z "${use_unified_batch_sampler}" ] && use_unified_batch_sampler=True
[ -z "${rescale_loss_with_std}" ] && rescale_loss_with_std=True
[ -z "${fp16}" ] && fp16=False

#[ -z "${loadcheck_path}" ] && loadcheck_path="/data/peiran/blob/hai1data/sfm/pfmexp/output/psmv1_vt_v3/checkpoints/global_step48063/mp_rank_00_model_states.pt"
#[ -z "${save_dir}" ] && save_dir="/home/peiran/expresult/psmexp/output/psmv1_vt_v3/"
[ -z "${loadcheck_path}" ] && loadcheck_path="/casp/jianwzhu/workspace/SFM_Evaluation/run_sfm/sfmblob/pfmexp/output/psmv1_vt_v3/checkpoints/global_step48063/mp_rank_00_model_states.pt"
[ -z "${save_dir}" ] && save_dir="/casp/jianwzhu/workspace/SFM_Evaluation/run_sfm/sfmblob/psm-checkpoints/"
[ -z "${dataset_name}" ] && dataset_name="."
[ -z "${add_3d}" ] && add_3d=true
[ -z "${no_2d}" ] && no_2d=false
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=0

[ -z "${wandb_group}" ] && wandb_group=psm_dev
[ -z "${wandb_team}" ] && wandb_team=ai4s-sfm
[ -z "${wandb_project}" ] && wandb_project=psm_dev
[ -z "${wandb_key}" ] && wandb_key=local-094f941ede8eda7a00c307f50595f054be5382f7

[ -z "${launcher}" ] && launcher="openmpi"
[ -z "${hostfile}" ] && hostfile="/job/hostfile"
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62347
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${equivar_vec_init}" ] && equivar_vec_init="RELATIVE_POS"
[ -z "${pbc_cutoff}" ] && pbc_cutoff=20.0
[ -z "${pbc_expanded_num_cell_per_direction}" ] && pbc_expanded_num_cell_per_direction=5
[ -z "${pbc_expanded_token_cutoff}" ] && pbc_expanded_token_cutoff=512
[ -z "${pbc_multigraph_cutoff}" ] && pbc_multigraph_cutoff=5.0
[ -z "${pbc_use_local_attention}" ] && pbc_use_local_attention=True
[ -z "${diffusion_noise_std}" ] && diffusion_noise_std=10.0
[ -z "${diffusion_mode}" ] && diffusion_mode=epsilon

[ -z "${diff_init_lattice_size}" ] && diff_init_lattice_size=10.0
[ -z "${diffusion_sampling}" ] && diffusion_sampling="ddpm"
[ -z "${num_timesteps}" ] && num_timesteps=5000
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=1e-7
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=2e-3
[ -z "${ddpm_schedule}" ] && ddpm_schedule=sigmoid

[ -z "${equivar_use_linear_bias}" ] && equivar_use_linear_bias=False
[ -z "${equivar_use_attention_bias}" ] && equivar_use_attention_bias=False


echo -e "\n\n"
echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
n_gpu=1
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

#wandb login --relogin --host=https://microsoft-research.wandb.io $wandb_key
#export WANDB_API_KEY=$wandb_key

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node 1 \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
  fi
fi

echo "DISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}"
#   num_attention_heads=$num_head \

torchrun $DISTRIBUTED_ARGS sfm/tasks/psm/generate_psm_protein.py \
          --config-name=config_psm.yaml \
          backbone_config=graphormer \
          backbone=vanillatransformer \
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
          infer=True \
          mask_ratio=$mask_ratio \
          noise_scale=$noise_scale \
          num_pred_attn_layer=$num_pred_attn_layer \
          d_tilde=$d_tilde \
          strategy=$strategy \
          max_lr=$max_lr \
          diffusion_mode=\"$diffusion_mode\" \
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
          wandb=True wandb_group=$wandb_group wandb_team=$wandb_team wandb_project=$wandb_project \
          fasta_list=$fasta_list \
          output_dir=$output_dir \
