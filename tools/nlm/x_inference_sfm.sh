#! /bin/bash
# export model_name="mixtral_8x7b"
# export base_model_root="/home/v-zekunguo/nlm/Mixtral-8x7B-v0.1/"
# export model_ckpt_home="/home/v-zekunguo/nlm/shufxi/nlm/8x7b/stageB_pp8_acc16_total1536_12m_bsz/global_step32000/"
# export aux_home="/home/v-zekunguo/blob/shufxi/data/scigpt"
# export temperature=0.75
# export output_dir="/home/v-zekunguo/nlm/zekun/instruct/8x7b/uncondition_less/test"
# # - export input_file="/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617.test"
# export input_file="/home/v-zekunguo/nlm/zekun/data/scidata/uncondition"
# [ -z "${model_name}" ] && model_name="llama3_8b"
# [ -z "${base_model_root}" ] && base_model_root="/home/v-zekunguo/nlm/llama/Meta-Llama-3-8B"
# # [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo/nlm/zekun/output/base1b/150B_G64_512/global_step44960"
# # [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo//nlm/kaiyuan/results/nlm/inst/inst_0621_bsz256_lr2e5_0624/global_step89920"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo//nlm/zekun/output/8b/t2df_bs256_lr2e5/global_step3225"

# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo/nlm/zekun/output/8b/alphadrug_bs256_lr2e5/global_step10000"
[ -z "${aux_home}" ] && aux_home="/home/v-zekunguo/blob/shufxi/data/scigpt/"

# [ -z "${model_name}" ] && model_name="llama2_7b"
# [ -z "${base_model_root}" ] && base_model_root="/home/v-zekunguo/blob/lihe/hai1/ds_dataset/llama2/llama-2-7b"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo/blob/yinxia/scigpt/7bv3/unifyall_TamGenv2_t2df/global_step2150"
# [ -z "${aux_home}" ] && aux_home="/home/lihe/mlla/shufxi/data/scigpt/"

# [ -z "${model_name}" ] && model_name="llama3_8b"
# [ -z "${base_model_root}" ] && base_model_root="/home/lihe/sfmdataeastus2_nlm/llama/Meta-Llama-3-8B/original/"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/lihe/sfmdataeastus2_nlm/kaiyuan/results/nlm/inst/inst_tuning_full_bsz128_lr5e-5_0616/global_step54000/"
# [ -z "${aux_home}" ] && aux_home=""

# [ -z "${model_name}" ] && model_name="mixtral_8x7b"
# [ -z "${base_model_root}" ] && base_model_root="/scratch/workspace/nlm/Mixtral-8x7B-v0.1/"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/scratch/workspace/nlm/sfmdataeastus2_nlm/shufxi/nlm/8x7b/inst/20240611215447/global_step33216/"
# [ -z "${aux_home}" ] && aux_home="/tmp/nlm_rank"

[ -z "${model_name}" ] && model_name="base1b"
[ -z "${base_model_root}" ] && base_model_root="/home/v-zekunguo/nlm/llama/Meta-Llama-3-8B"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo/nlm/zekun/output/1b/chembl_t2d_G256_bs256_lr2e5/global_step21627"
# [ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo/nlm/zekun/output/1b/table_nest_mask_protein/global_step303"
[ -z "${model_ckpt_home}" ] && model_ckpt_home="/home/v-zekunguo/nlm/zekun/output/1b/table_base_mask_protein/global_step6746"
# [ -z "${input_file}" ] && input_file="/home/v-zekunguo/nlm/lihe/data/generated/sfmdata.prot.test.sampled30.tsv"
[ -z "${input_file}" ] && input_file="/home/v-zekunguo/nlm/zekun/data/scidata/uncondition_16/material.txt"
# [ -z "${input_file}" ] && input_file="/home/v-zekunguo/blob/v-houtianzhu/process_tsv/current_result/test_table.txt"
# [ -z "${input_file}" ] && input_file="/home/v-zekunguo/blob/v-houtianzhu/process_tsv/current_result/table_test.json.txt"
[ -z "${output_dir}" ] && output_dir="/home/v-zekunguo/logs/1b/base_table"

[ -z "${command}" ] && command="inference"
# [ -z "${command}" ] && command="generate"
[ -z "${n_seq}" ] && n_seq=128
[ -z "${entity}" ] && entity=protein
[ -z "${temperature}" ] && temperature=0.7
echo "input_file: ${input_file}"
echo "output_dir: ${output_dir}"
echo "model_name: ${model_name}"
echo "base_model_root: ${base_model_root}"
echo "model_ckpt_home: ${model_ckpt_home}"
echo "aux_home: ${aux_home}"

[ -z "${launcher}" ] && launcher='openmpi'
[ -z "${hostfile}" ] && hostfile='/job/hostfile'
[ -z "${MASTER_PORT}" ] && MASTER_PORT=62349
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

echo "==================================MP==========================================="

[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
# [ -z "${n_gpu}" ] && n_gpu=3

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
# export CUDA_VISIBLE_DEVICES="1,2,3"
torchrun $DISTRIBUTED_ARGS tools/nlm/x_inference_sfm.py \
    --model_name ${model_name} \
    --input_file ${input_file} \
    --output_dir ${output_dir} \
    --base_model_root ${base_model_root} \
    --model_ckpt_home ${model_ckpt_home} \
    --aux_home ${aux_home} \
    --command ${command} \
    --temperature ${temperature} \
    --n_seq ${n_seq} --entity ${entity}

echo "done"
# sleep inf
# sleep inf
