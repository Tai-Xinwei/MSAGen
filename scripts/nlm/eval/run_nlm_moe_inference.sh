#!/bin/bash

# Set the path to the checkpoint directory
MIXTRAL_PATH='/nlm/Mixtral-8x7B-v0.1'
NLM_PATH='/nlm/shufxi/nlm/8x7b/inst/20240611215447/global_step33216'
LOCAL_PATH='/dev/shm/nlm'

# Set the path to the input directory where the test files are located
INPUT_DIR='/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617.test'

# Set the path to the output directory where the inference results will be saved
OUTPUT_DIR='/home/shufxi/workspace/SFM_NLM_resources/all_infer_results/8x7b_global_step33216_infer_results'

mkdir -p $OUTPUT_DIR

# Execute the Python module with the specified arguments

python3 sfm/tasks/nlm/eval/run_nlm_moe_inference.py \
    --mixtral_path "$MIXTRAL_PATH" \
    --nlm_path "$NLM_PATH" \
    --local_path "$LOCAL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
