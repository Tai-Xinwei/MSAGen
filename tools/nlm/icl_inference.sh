#! /bin/bash
[ -z "${model_name}" ] && model_name="llama3_1b"
[ -z "${input_file}" ] && input_file="/scratch/workspace/nlm/protein/generated/sfmdata.prot.test.sampled30.tsv"
[ -z "${output_dir}" ] && output_dir="/scratch/workspace/nlm/protein/output/${model_name}"

[ -z "${command}" ] && command="inference"
[ -z "${n_seq}" ] && n_seq=128
[ -z "${entity}" ] && entity=product
[ -z "${max_new_tokens}" ] && max_new_tokens=8192

echo "input_file: ${input_file}"
echo "output_dir: ${output_dir}"
echo "model_name: ${model_name}"
echo "base_model_root: ${base_model_root}"
echo "model_ckpt_home: ${model_ckpt_home}"
echo "aux_home: ${aux_home}"
echo "max_new_tokens: ${max_new_tokens}"

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

mkdir -p ${output_dir}
torchrun $DISTRIBUTED_ARGS tools/nlm/icl_reactant_inference.py \
    --model_name ${model_name} \
    --input_file ${input_file} \
    --output_dir ${output_dir} \
    --base_model_root ${base_model_root} \
    --model_ckpt_home ${model_ckpt_home} \
    --aux_home ${aux_home} \
    --command ${command} \
    --n_seq ${n_seq} --entity ${entity} --max_new_tokens ${max_new_tokens}

echo "done"
