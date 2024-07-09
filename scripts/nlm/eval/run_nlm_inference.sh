#!/bin/bash

# Set the path to the checkpoint directory
[ -z "${CKPT_HOME}" ] && CKPT_HOME='/home/yeqibai/mount/ml_la/yinxia/scigpt/7bv3/unifyall_v2_full_run1/global_step17984'

# Tokenizers and BPE model files
[ -z "${TOKENIZER_HOME}" ] && TOKENIZER_HOME='/home/yeqibai/mount/ml_la/yeqibai/warehouse/llama/llama-2-7b'
[ -z "${PROT_SPM_PATH}" ] && PROT_SPM_PATH='/home/yeqibai/mount/ml_la/yeqibai/warehouse/scigpt/ur50bpe/bpe'
[ -z "${DNA_SPM_PATH}" ] && DNA_SPM_PATH='/home/yeqibai/mount/ml_la/yeqibai/warehouse/scigpt/dnabpe/bpe'
[ -z "${RNA_SPM_PATH}" ] && RNA_SPM_PATH='/home/yeqibai/mount/ml_la/yeqibai/warehouse/scigpt/rnabpe/bpe'

# Set the path to the input directory where the test files are located
[ -z "${INPUT_DIR}" ] && INPUT_DIR='/home/yeqibai/mount/ml_la/yeqibai/warehouse/nlm_data/instruct/molecules_test/'

# Set the path to the output directory where the inference results will be saved
[ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR='/home/yeqibai/workspace/SFM_NLM_resources/all_infer_results/unifyall_v2_full_run1_global_step17984_infer_results'

echo "Checkpoint Home: $CKPT_HOME"
echo "Tokenizer Home: $TOKENIZER_HOME"
echo "Protein SPM Path: $PROT_SPM_PATH"
echo "DNA SPM Path: $DNA_SPM_PATH"
echo "RNA SPM Path: $RNA_SPM_PATH"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Execute the Python module with the specified arguments
python3 sfm/tasks/nlm/eval/run_nlm_inference.py \
    --ckpt_home "$CKPT_HOME" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer_home "$TOKENIZER_HOME" \
    --prot_spm_path "$PROT_SPM_PATH" \
    --dna_spm_path "$DNA_SPM_PATH" \
    --rna_spm_path "$RNA_SPM_PATH"
