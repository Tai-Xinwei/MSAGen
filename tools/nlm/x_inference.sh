#! /bin/bash

[ -z "${model_name}" ] && model_name="llama2_7b"
[ -z "${base_model_root}" ] && base_model_root="/home/lihe/mlla/lihe/hai1/ds_dataset/llama2/llama-2-7b"
[ -z "${model_ckpt_home}" ] && model_ckpt_home="/scratch/workspace/mlla/yinxia/scigpt/7bv3/unifyall_full_run1/global_step22144"
[ -z "${aux_home}" ] && aux_home="/home/lihe/mlla/shufxi/data/scigpt/"

# [ -z "${model_name}" ] && model_name="llama2_7b"
# [ -z "${base_model_root}" ] && base_model_root="/home/lihe/mlla/lihe/hai1/ds_dataset/llama2/llama-2-7b"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/scratch/workspace/mlla/yinxia/scigpt/7bv3/unifyall_v2_full_run1/global_step17984"
# [ -z "${aux_home}" ] && aux_home="/home/lihe/mlla/shufxi/data/scigpt/"

# [ -z "${model_name}" ] && model_name="llama3_8b"
# [ -z "${base_model_root}" ] && base_model_root="/home/lihe/sfmdataeastus2_nlm/llama/Meta-Llama-3-8B/original/"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/lihe/sfmdataeastus2_nlm/kaiyuan/results/nlm/inst/inst_tuning_full_bsz128_lr5e-5_0616/global_step54000/"
# [ -z "${aux_home}" ] && aux_home=""

# [ -z "${model_name}" ] && model_name="mixtral_8x7b"
# [ -z "${base_model_root}" ] && base_model_root="/scratch/workspace/nlm/Mixtral-8x7B-v0.1/"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/scratch/workspace/nlm/sfmdataeastus2_nlm/shufxi/nlm/8x7b/inst/20240611215447/global_step33216/"
# [ -z "${aux_home}" ] && aux_home="/tmp/nlm_rank"

[ -z "${input_file}" ] && input_file="/scratch/workspace/nlm/protein/generated/sfmdata.prot.test.sampled30.tsv"
[ -z "${output_dir}" ] && output_dir="/scratch/workspace/nlm/protein/output/${model_name}"

[ -z "${command}" ] && command="inference"
[ -z "${n_seq}" ] && n_seq=128
[ -z "${entity}" ] && entity=protein

echo "input_file: ${input_file}"
echo "output_dir: ${output_dir}"
echo "model_name: ${model_name}"
echo "base_model_root: ${base_model_root}"
echo "model_ckpt_home: ${model_ckpt_home}"
echo "aux_home: ${aux_home}"

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

torchrun $DISTRIBUTED_ARGS tools/nlm/x_inference.py \
    --model_name ${model_name} \
    --input_file ${input_file} \
    --output_dir ${output_dir} \
    --base_model_root ${base_model_root} \
    --model_ckpt_home ${model_ckpt_home} \
    --aux_home ${aux_home} \
    --command ${command} \
    --n_seq ${n_seq} --entity ${entity}

echo "done"
