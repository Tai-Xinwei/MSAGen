#!/bin/bash

# Set the paths
[ -z "${MIXTRAL_PATH}" ] && MIXTRAL_PATH='/home/v-yinzhezhou/new_branch_SFM/SFM_framework/eval_testing_data/nlm_moe_inference/mixtral/Mixtral-8x7B-v0.1'
[ -z "${NLM_PATH}" ] && NLM_PATH='/home/v-yinzhezhou/new_branch_SFM/SFM_framework/eval_testing_data/nlm_moe_inference/nlm'
[ -z "${LOCAL_PATH}" ] && LOCAL_PATH='/dev/shm/nlm'
[ -z "${INPUT_DIR}" ] && INPUT_DIR='/home/v-yinzhezhou/new_branch_SFM/SFM_framework/eval_testing_data/nlm_moe_inference/input_data'
[ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR='/home/v-yinzhezhou/new_branch_SFM/SFM_framework/eval_testing_data/evaluate_small_molecule/output_data'

# Your script logic here
echo "Mixtral Path: $MIXTRAL_PATH"
echo "NLM Path: $NLM_PATH"
echo "Local Path: $LOCAL_PATH"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"


# Execute the Python module with the specified arguments
python3 sfm/tasks/nlm/eval/run_nlm_moe_inference.py \
    --mixtral_path "$MIXTRAL_PATH" \
    --nlm_path "$NLM_PATH" \
    --local_path "$LOCAL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
