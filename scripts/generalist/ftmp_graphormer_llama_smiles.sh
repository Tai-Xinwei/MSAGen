#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# molecule model parameters
[ -z "${layers}" ] && layers=24
[ -z "${num_pred_attn_layer}" ] && num_pred_attn_layer=4
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

# molecule model training parameters
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.0
[ -z "${attn_dropout}" ] && attn_dropout=0.0
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln=true
[ -z "${droppath_prob}" ] && droppath_prob=0.0
if [[ "${sandwich_ln}" == "true" ]]; then
  sandwich_ln="--sandwich_ln"
else
  sandwich_ln=""
fi

# generalist model parameters
if [[ "${fused_graphormer_llama}" == "true" ]]; then
  fused_graphormer_llama="--fused_graphormer_llama"
else
  fused_graphormer_llama=""
fi
if [[ "${add_mol_attn_bias_in_llama}" == "true" ]]; then
  add_mol_attn_bias_in_llama="--add_mol_attn_bias_in_llama"
else
  add_mol_attn_bias_in_llama=""
fi
if [[ "${mol_attn_bias_in_llama_layerwise}" == "true" ]]; then
  mol_attn_bias_in_llama_layerwise="--mol_attn_bias_in_llama_layerwise"
else
  mol_attn_bias_in_llama_layerwise=""
fi
[ -z "${path_edge_cutoff}" ] && path_edge_cutoff=20

# general training parameters
[ -z "${d_tilde}" ] && d_tilde=1
[ -z "${max_lr}" ] && max_lr=2e-5
[ -z "${total_num_steps}" ] && total_num_steps=10000
[ -z "${warmup_num_steps}" ] && warmup_num_steps=600
[ -z "${seed}" ] && seed=12345

# generalist dataset settings
[ -z "${data_path}" ] && data_path='/mnt/shiyu/dataset/chemical-copilot-special-token/'
# [ -z "${data_path}" ] && data_path='/home/peiran/mnt/mntsfm2/data/chemical-copilot-special-token/'
# [ -z "${data_path}" ] && data_path='/mnt/chemical-copilot-special-token'
# [ -z "${data_path}" ] && data_path='/mnt/shiyu/dataset/chemical-copilot-special-token'

[ -z "${dataset_names}" ] && dataset_names='mol-instruction-mol-desc'
[ -z "${dataset_splits}" ] && dataset_splits='clean'
# [ -z "${dataset_names}" ] && dataset_names='mol-instruction-mol-desc,chebi,functional-group,func-group-list-and-desc,chemcop-instruction,tdc/LD50_Zhu,tdc/kcnq2_potassium_channel_butkiewicz,tdc/Skin_Reaction,tdc/HIV,tdc/CYP3A4_Veith,tdc/CYP1A2_Veith,tdc/hERG_Karim,tdc/PAMPA_NCATS,tdc/hERG,tdc/CYP2C9_Substrate_CarbonMangels,tdc/HydrationFreeEnergy_FreeSolv,tdc/m1_muscarinic_receptor_agonists_butkiewicz,tdc/Bioavailability_Ma,tdc/m1_muscarinic_receptor_antagonists_butkiewicz,tdc/DILI,tdc/potassium_ion_channel_kir2.1_butkiewicz,tdc/CYP2C9_Veith,tdc/SARSCoV2_3CLPro_Diamond,tdc/Clearance_Hepatocyte_AZ,tdc/choline_transporter_butkiewicz,tdc/Half_Life_Obach,tdc/Lipophilicity_AstraZeneca,tdc/cav3_t-type_calcium_channels_butkiewicz,tdc/SARSCoV2_Vitro_Touret,tdc/Caco2_Wang,tdc/VDss_Lombardo,tdc/PPBR_AZ,tdc/Solubility_AqSolDB,tdc/tyrosyl-dna_phosphodiesterase_butkiewicz,tdc/Carcinogens_Lagunin,tdc/Pgp_Broccatelli,tdc/CYP2C19_Veith,tdc/CYP3A4_Substrate_CarbonMangels,tdc/CYP2D6_Substrate_CarbonMangels,tdc/serine_threonine_kinase_33_butkiewicz,tdc/orexin1_receptor_butkiewicz,tdc/AMES,tdc/CYP2D6_Veith,tdc/Tox21/NR-AR,tdc/Tox21/NR-PPAR-gamma,tdc/Tox21/NR-AR-LBD,tdc/Tox21/NR-Aromatase,tdc/Tox21/SR-MMP,tdc/Tox21/NR-AhR,tdc/Tox21/SR-HSE,tdc/Tox21/NR-ER,tdc/Tox21/SR-ARE,tdc/Tox21/NR-ER-LBD,tdc/Tox21/SR-p53,tdc/Tox21/SR-ATAD5,tdc/herg_central/hERG_inhib,tdc/herg_central/hERG_at_1uM,tdc/herg_central/hERG_at_10uM,tdc/USPTO_Yields,tdc/Buchwald-Hartwig'
# [ -z "${dataset_splits}" ] && dataset_splits='clean,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all'
# [ -z "${dataset_ratios}" ] && dataset_ratios='5.0,5.0,5.0,5.0,1.0,16.93,0.41,309.60,3.04,10.14,9.52,9.30,61.46,191.20,187.27,194.55,2.02,195.31,2.02,263.85,0.41,10.34,142.05,103.09,0.41,187.97,29.76,1.24,84.25,137.36,110.62,44.82,12.52,0.37,446.43,102.77,10.34,186.92,187.97,0.39,0.57,17.18,9.52,17.21,19.38,18.50,21.47,21.51,19.09,19.33,20.19,21.43,17.97,18.45,17.67,0.41,0.41,0.41,0.15,31.61'
# [ -z "${dataset_names}" ] && dataset_names='chebi'
# [ -z "${dataset_splits}" ] && dataset_splits='all'
# [ -z "${dataset_ratios}" ] && dataset_ratios='1.0'
# [ -z "${dataset_names}" ] && dataset_names='chemcop-instruction'
# [ -z "${dataset_splits}" ] && dataset_splits='all'
[ -z "${dataset_ratios}" ] && dataset_ratios='1.0'

[ -z "${pool_mode}" ] && pool_mode='full'
[ -z "${embedding_length}" ] && embedding_length=20
[ -z "${model_max_length}" ] && model_max_length=512

# checkpoint and log settings
[ -z "${save_dir}" ] && save_dir='/mnt/shiyu/checkpoints/llama2-local-debug'
# [ -z "${save_dir}" ] && save_dir='/home/peiran/FMproj/output'
[ -z "${save_batch_interval}" ] && save_batch_interval=5000
[ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/mnt/shiyu/models/graphormer_ckpts/checkpoint7_new.pt"
[ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/mnt/shiyu/models/converted/llama-2-7b"
# [ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/home/peiran/FMproj/DiffTM100M/checkpoint7_new.pt"
# [ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/mnt/peiran/pretrain56w/global_step560000"
# [ -z "${loadmfmcheck_path}" ] && loadmfmcheck_path="/home/peiran/FMproj/DiffTM100M/tp"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/home/peiran/FMproj/llama2/llama-2-7b"
# [ -z "${llm_model_name_or_path}" ] && llm_model_name_or_path="/mnt/peiran/llama-2-70b"
[ -z "${finetune_from_checkpoint_dir}" ] && finetune_from_checkpoint_dir=""
[ -z "${finetune_from_checkpoint_id}" ] && finetune_from_checkpoint_id=""
[ -z "${wandb_key}" ] && wandb_key=5d03b7a46d10f86ff45c4aedc570660a523edc0b
[ -z "${wandb_project_name}" ] && wandb_project_name="chemical-generalist"
if [[ $finetune_from_checkpoint_dir != "" ]]; then
  finetune_from_checkpoint_dir="--finetune_from_checkpoint_dir ${finetune_from_checkpoint_dir}"
  finetune_from_checkpoint_id="--finetune_from_checkpoint_id ${finetune_from_checkpoint_id}"
fi

# training parallelism
[ -z "${pipeline_model_parallel_size}" ] && pipeline_model_parallel_size=1
[ -z "${tensor_model_parallel_size}" ] && tensor_model_parallel_size=4
[ -z "${strategy}" ] && strategy=ThreeD
# determine zero strategy in DeepSpeed
if [[ "${strategy}" == "Zero1" || "${strategy}" == "Pipeline" || "${strategy}" == "ThreeD" ]]; then
  zero_strategy=1
elif [[ ${strategy} == "Zero2" ]]; then
  zero_strategy=2
elif [[ ${strategy} == "Zero3" ]]; then
  zero_strategy=3
fi
[ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="LlamaDecoderLayer"
[ -z "${pp_part_list}" ] && pp_part_list="[0, 61]"
# [ -z "${pp_partition_layer_name}" ] && pp_partition_layer_name="manual"
# [ -z "${pp_part_list}" ] && pp_part_list="[0, 43, 70, 99, 127]"
[ -z "${unfreeze_param_list}" ] && unfreeze_param_list="mol_adaptor,mol_rep_layernorm,word_embeddings"

# training parameters for generalist
[ -z "${micro_batch_size}" ] && micro_batch_size=1
[ -z "${global_batch_size}" ] && global_batch_size=16
[ -z "${max_position_embeddings}" ] && max_position_embeddings=2048
[ -z "${llm_hidden_size}" ] && llm_hidden_size=4096

# prepare megatron args
if [[ "${strategy}" == "ThreeD" ]]; then
  MEGATRON_ARGS="--micro-batch-size $micro_batch_size --global-batch-size $global_batch_size \
    --num-layers $layers --hidden-size $llm_hidden_size --seq-length $max_position_embeddings \
    --max-position-embeddings $max_position_embeddings --num-attention-heads $num_head \
    --seq-length $max_position_embeddings --disable-bias-linear --no-position-embedding --no-query-key-layer-scaling"
else
  MEGATRON_ARGS=""
fi

# default env variables for distributed training
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=0


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


echo -e "\n\n"
echo "===================================== start of parameters ======================================"
echo "===================================== molecule model parameters ====================================="
echo "n_layers: ${layers}"
echo "num_pred_attn_layer: ${num_pred_attn_layer}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "num_3d_bias_kernel: ${num_3d_bias_kernel}"

echo "===================================== molecule model training parameters ====================================="
echo "dropout: ${dropout}"
echo "act_dropout: ${act_dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "weight_decay: ${weight_decay}"
echo "sandwich_ln: ${sandwich_ln}"
echo "droppath_prob: ${droppath_prob}"

echo "===================================== general training parameters ====================================="
echo "d_tilde: ${d_tilde}"
echo "max_lr: ${max_lr}"
echo "total_num_steps: ${total_num_steps}"
echo "warmup_num_steps: ${warmup_num_steps}"
echo "seed: ${seed}"


echo "===================================== generalist dataset settings ====================================="
echo "data_path: ${data_path}"
echo "dataset_names: ${dataset_names}"
echo "dataset_splits: ${dataset_splits}"
echo "dataset_ratios: ${dataset_ratios}"
echo "pool_mode: ${pool_mode}"
echo "embedding_length: ${embedding_length}"
echo "model_max_length: ${model_max_length}"

echo "===================================== checkpoint and log settings ====================================="
echo "save_dir: ${save_dir}"
echo "save_batch_interval: ${save_batch_interval}"
echo "loadmfmcheck_path: ${loadmfmcheck_path}"
echo "llm_model_name_or_path: ${llm_model_name_or_path}"
echo "wandb_key: ${wandb_key}"
echo "wandb_project_name: ${wandb_project_name}"

echo "===================================== training parallelism ====================================="
echo "pipeline_model_parallel_size: ${pipeline_model_parallel_size}"
echo "tensor_model_parallel_size: ${tensor_model_parallel_size}"
echo "strategy: ${strategy}"
echo "pp_partition_layer_name: ${pp_partition_layer_name}"
echo "pp_part_list: ${pp_part_list}"
echo "unfreeze_param_list: ${unfreeze_param_list}"

echo "===================================== training parameters for generalist ====================================="
echo "micro_batch_size: ${micro_batch_size}"
echo "global_batch_size: ${global_batch_size}"
echo "max_position_embeddings: ${max_position_embeddings}"
echo "llm_hidden_size: ${llm_hidden_size}"

echo "===================================== end of parameters ======================================"
echo -e "\n\n"


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

wandb login --relogin ${wandb_key}

mkdir -p $save_dir
DS_CONFIG=$save_dir/deepspeed.json

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $global_batch_size,
  "train_micro_batch_size_per_gpu": $micro_batch_size,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": $zero_strategy
  },
  "fp16": {
    "enabled": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0000,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.0
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_type": "linear",
      "total_num_steps": $total_num_steps,
      "warmup_max_lr": $max_lr,
      "warmup_num_steps": $warmup_num_steps
    }
  },
  "wandb": {
    "enabled": true,
    "project": "$wandb_project_name"
  }
}
EOT


if [[ $OMPI_COMM_WORLD_SIZE > 1 ]]; then
  cp sfm/utils/barrier.py . && touch READY && python -u barrier.py $OMPI_COMM_WORLD_SIZE $OMPI_COMM_WORLD_RANK
fi

torchrun $DISTRIBUTED_ARGS sfm/tasks/generalist/ft_graphormer_llama_inst.py \
          --encoder_layers $layers \
          --num_pred_attn_layer $num_pred_attn_layer \
          --encoder_embed_dim $hidden_size \
          --encoder_ffn_embed_dim $ffn_size \
          --encoder_attention_heads $num_head \
          --num_3d_bias_kernel $num_3d_bias_kernel \
          --dropout $dropout \
          --act_dropout $act_dropout \
          --attn_dropout $attn_dropout \
          --weight_decay $weight_decay \
          ${sandwich_ln} \
          --droppath_prob $droppath_prob \
          --d_tilde $d_tilde \
          --max_lr $max_lr \
          --total_num_steps $total_num_steps \
          --warmup_num_steps $warmup_num_steps \
          --seed $seed \
          --data_path $data_path \
          --dataset_names $dataset_names \
          --dataset_splits $dataset_splits \
          --dataset_ratios $dataset_ratios \
          --pool_mode $pool_mode \
          --embedding_length $embedding_length \
          --model_max_length $model_max_length \
          --save_dir $save_dir \
          --save_batch_interval $save_batch_interval \
          --loadmfmcheck_path $loadmfmcheck_path \
          --llm_model_name_or_path $llm_model_name_or_path \
          ${finetune_from_checkpoint_dir} \
          ${finetune_from_checkpoint_id} \
          --pipeline-model-parallel-size $pipeline_model_parallel_size \
          --tensor-model-parallel-size $tensor_model_parallel_size \
          --strategy $strategy \
          --pp_part_list "${pp_part_list}" \
          --pp_partition_layer_name $pp_partition_layer_name \
          --unfreeze_param_list $unfreeze_param_list \
          --ft --load_ckpt \
          --fp16 \
          --deepspeed_config=$DS_CONFIG \
          ${MEGATRON_ARGS} \
          --num_data_loading_workers ${num_data_loading_workers} \
          --skip_num_datasets "${skip_num_datasets}" \
          --use_global_padding \
          --multi_hop_max_dist 7 \
          --max_num_mol_per_sample  3 \
          --molecule_max_size 128
          # --fused_graphormer_llama \
          # --use_pbc \
          # --add_3d \
