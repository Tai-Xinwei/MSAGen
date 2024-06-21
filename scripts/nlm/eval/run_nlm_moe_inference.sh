#!/bin/bash

# Set the path to the checkpoint directory
MIXTRAL_PATH='/home/yeqibai/mount/nlm/Mixtral-8x7B-v0.1'
NLM_PATH='/home/yeqibai/mount/nlm/shufxi/nlm/8x7b/inst/20240611215447/global_step33216'
LOCAL_PATH='/scratch/tmp'

# Set the path to the input directory where the test files are located
INPUT_DIR='/home/yeqibai/mount/ml_la/yeqibai/warehouse/nlm_data/instruct/molecules_test/'

# Set the path to the output directory where the inference results will be saved
OUTPUT_DIR='/home/yeqibai/workspace/SFM_NLM_resources/all_infer_results/8x7b_global_step33216_infer_results'

# Execute the Python module with the specified arguments
python3 run_nlm_moe_inference.py \
    --mixtral_path "$MIXTRAL_PATH" \
    --nlm_path "$NLM_PATH" \
    --local_path "$LOCAL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
