#!/bin/bash

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62346
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

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

export MODEL_PARALLEL_SIZE=${n_gpu}

# Set the paths
[ -z "${MODEL_NAME}" ] && MODEL_NAME='mixtral_8x7b-dist'
[ -z "${BASE_MODEL_ROOT}" ] && BASE_MODEL_ROOT='/home/yeqi/mount/nlm/Mixtral-8x7B-v0.1/'
[ -z "${AUX_HOME}" ] && AUX_HOME='/data/yeqi/cache/nlm_moe'
[ -z "${MODEL_CKPT_HOME}" ] && MODEL_CKPT_HOME='/home/yeqi/mount/nlm/shufxi/nlm/8x7b/inst/20240611215447/global_step33216'

# Your script logic here
echo "Model Name: $MODEL_NAME"
echo "Base Path: $BASE_MODEL_ROOT"
echo "Local Path: $AUX_HOME"
echo "NLM Path: $MODEL_CKPT_HOME"


# Execute the Python module with the specified arguments
torchrun $DISTRIBUTED_ARGS tools/nlm/x_inference.py \
    --model_name ${MODEL_NAME} \
    --base_model_root ${BASE_MODEL_ROOT} \
    --model_ckpt_home ${MODEL_CKPT_HOME} \
    --aux_home ${AUX_HOME} \
    --command 'measure_infer_speed'
