#!/user/bin/env bash
ulimit -c unlimited
nvidia-smi

if [ -z "${mixtral_path}" ]; then
    echo "mixtral_path is not set"
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "model_path is not set"
    exit 1
fi

if [ -z "${output_path}" ]; then
    echo "output_path is not set"
    exit 1
fi

[ -z "${n_seq}" ] && n_seq=128
[ -z "${entity}" ] && entity="protein"

output_path="${output_path}/pred_${LOCAL_RANK}.txt"
local_path="/tmp/nlm_rank${LOCAL_RANK}"
offload_path="/tmp/moe_rank${LOCAL_RANK}"

mkdir -p "${local_path}"
mkdir -p "${offload_path}"

echo "Start generation on rank $LOCAL_RANK, output_path: $output_path"
export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}

python sfm/tasks/nlm/generate_entity_moe.py  \
    --mixtral_path "${mixtral_path}" \
    --nlm_path "${model_path}" \
    --local_path "${local_path}" \
    --offload_path "${offload_path}" \
    --output_path "${output_path}" \
    --n_seq "${n_seq}" \
    --entity "${entity}" \

echo "example output:"
head -n 10 "${output_path}"
echo "Done"
