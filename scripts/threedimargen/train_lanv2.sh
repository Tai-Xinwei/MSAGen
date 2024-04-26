# default env variables for distributed training
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_LOCAL_RANK}" ] && OMPI_COMM_WORLD_LOCAL_RANK=0
[ -z "${GPUS}" ] && GPUS=$(nvidia-smi -L | wc -l)

export OMPI_COMM_WORLD_RANK=$OMPI_COMM_WORLD_RANK
export OMPI_COMM_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

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

torchrun $DISTRIBUTED_ARGS sfm/tasks/threedimargen/train_threedimargenlan.py \
    --dict_path sfm/data/threedimargen_data/dict_lan_v2.txt \
    --train_data_path /hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train.jsonl \
    --valid_data_path /hai1/SFM/threedimargen/data/materials_data/carbon-24_perov-5_mp-20_val.jsonl \
    --save_dir /hai1/SFM/threedimargen/outputs/${NAME} \
    --model_type threedimargen \
    --tokenizer lanv2 \
    --fp16 \
    --strategy DDP \
    --total_num_epochs 50 \
    --warmup_num_steps 8000 \
    --max_lr 1e-4 \
    --weight_decay 0.1 \
    --train_batch_size 256 \
    --val_batch_size 256 \
    --gradient_accumulation_steps 4 \
    --log_interval 100 \
    #--niggli_reduced \
