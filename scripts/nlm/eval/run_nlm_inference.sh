#!/bin/bash

# Set the path to the checkpoint directory
CKPT_HOME='/home/yeqibai/mount/ml_la/yinxia/scigpt/7bv3/unifyall_v2_full_run1/global_step17984'

# Tokenizers and BPE model files
TOKENIZER_HOME='/home/yeqibai/mount/ml_la/yeqibai/warehouse/llama/llama-2-7b'
PROT_SPM_PATH='/home/yeqibai/mount/ml_la/yeqibai/warehouse/scigpt/ur50bpe/bpe'
DNA_SPM_PATH='/home/yeqibai/mount/ml_la/yeqibai/warehouse/scigpt/dnabpe/bpe'
RNA_SPM_PATH='/home/yeqibai/mount/ml_la/yeqibai/warehouse/scigpt/rnabpe/bpe'

# Set the path to the input directory where the test files are located
INPUT_DIR='/home/yeqibai/mount/ml_la/yeqibai/warehouse/nlm_data/instruct/molecules_test/'

# Set the path to the output directory where the inference results will be saved
OUTPUT_DIR='/home/yeqibai/workspace/SFM_NLM_resources/all_infer_results/unifyall_v2_full_run1_global_step17984_infer_results'

# Execute the Python module with the specified arguments
python3 sfm/tasks/nlm/eval/run_nlm_inference.py \
    --ckpt_home "$CKPT_HOME" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer_home "$TOKENIZER_HOME" \
    --prot_spm_path "$PROT_SPM_PATH" \
    --dna_spm_path "$DNA_SPM_PATH" \
    --rna_spm_path "$RNA_SPM_PATH"
