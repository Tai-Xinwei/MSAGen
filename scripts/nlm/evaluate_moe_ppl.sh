#!/usr/bin/env bash
ulimit -c unlimited

nvidia-smi

# check parameters
if [ -z "${test_file_folder}" ]; then
    echo "test_file_folder is not set"
    exit 1
fi

if [ -z "${mixtral_path}" ]; then
    echo "mixtral_path is not set"
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "model_path is not set"
    exit 1
fi

if [ -z "${steps}" ]; then
    echo "steps is not set"
    exit 1
fi

ls -l "${test_file_folder}"/valid.*
data_path_list=()
for file in "${test_file_folder}"/valid.*; do
    if [ -f "$file" ]; then
        data_path_list+=("$file")
    fi
done

IFS=',' read -r -a step_list <<< "${steps}"
if [ ${#step_list[@]} -eq 1 ]; then
    echo "Found only one step: ${step_list[0]}"
    echo "Each rank will evaluate one test file"

    step=${step_list[0]}
    data_path_list_str=${data_path_list[LOCAL_RANK]}
else
    echo "Found multiple step_list:" "${step_list[@]}"
    echo "Each rank will evaluate one step"

    if ((LOCAL_RANK >= ${#step_list[@]} )); then
        step=${step_list[-1]}
    else
        step=${step_list[LOCAL_RANK]}
    fi

    data_path_list_str=$(IFS=,; echo "${data_path_list[*]}")
fi

echo "on rank $LOCAL_RANK, step is $step, data_path_list_str: ${data_path_list_str}"

local_path="/tmp/nlm_step${step}"
offload_path="/tmp/moe_step${step}"
mkdir -p "${local_path}"
mkdir -p "${offload_path}"


export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
set -x
python sfm/tasks/nlm/evaluate_moe_ppl.py \
    --mixtral_path "${mixtral_path}" \
    --nlm_path "${model_path}/global_step${step}" \
    --local_path "${local_path}" \
    --offload_path "${offload_path}" \
    --data_path_list "${data_path_list_str}"

echo "Done"
