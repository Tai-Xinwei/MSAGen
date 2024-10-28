# default env variables for distributed training
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=0
[ -z "${GPUS}" ] && GPUS=$(nvidia-smi -L | wc -l)

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  DISTRIBUTED_ARGS=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --master_port $MASTER_PORT"
  else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS \
                      --nnodes $OMPI_COMM_WORLD_SIZE \
                      --node_rank $OMPI_COMM_WORLD_RANK \
                      --master_addr $MASTER_ADDR"
  fi
fi


NAME="debug"
export WANDB_API_KEY="local-8904f1b2b7174cdb7c0bf0b8a2c3195c1889c537"
export WANDB_BASE_URL="https://microsoft-research.wandb.io"
export WANDB_TEAM=ai4s-sfm
export WANDB_PROJECT=${NAME}

torchrun $DISTRIBUTED_ARGS sfm/tasks/threedimargen/train_threedimargendiff.py \
    --dict_path sfm/data/threedimargen_data/dict.txt \
    --train_data_path /msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train.jsonl \
    --valid_data_path /msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/carbon-24_perov-5_mp-20_val.jsonl \
    --save_dir /msralaphilly2/ml-la/renqian/SFM/threedimargen/outputs/${NAME} \
    --total_num_epochs 50 \
    --warmup_num_steps 8000 \
    --max_lr 1e-4 \
    --weight_decay 0.1 \
    --max_sites 500 \
    --train_batch_size 256 \
    --val_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --log_interval 100 \
    --strategy DDP \
    --scale_coords 10 \
    --no_niggli_reduced
#/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train.jsonl \
