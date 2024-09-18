#!/bin/bash

# HF model name, DO NOT Modify
[ -z "${MIXTRAL_PATH}" ] && MIXTRAL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
# This is the HF-format NLM model ckpt
[ -z "${NLM_LOCAL_PATH}" ] && NLM_LOCAL_PATH="/data/yeqi/cache/nlm_moe"
# Please DO NOT Modify the following paths
[ -z "${INPUT_DIR}" ] && INPUT_DIR='./sfm/tasks/nlm/eval/inputs/nlm_data/science_test'
[ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR='./sfm/tasks/nlm/eval/outputs/all_infer_results/8x7b_step33216_revised_v2_vllm_infer_results'

# Your script logic here
echo "Mixtral Path: $MIXTRAL_PATH"
echo "NLM Local Path: $NLM_LOCAL_PATH"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"


# Execute the Python module with the specified arguments
python3 ./sfm/tasks/nlm/eval/run_nlm_moe_vllm_inference.py \
    --mixtral_path "$MIXTRAL_PATH" \
    --nlm_local_path "$NLM_LOCAL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
