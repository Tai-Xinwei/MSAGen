#!/bin/bash

for step in 89920
do
    CKPT_HOME=/home/t-kaiyuangao/sfmdata-container/kaiyuan/results/nlm/inst/inst_0621_bsz256_lr2e5_0624/global_step${step}
    INPUT_DIR=/home/t-kaiyuangao/ml-container/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617.test
    OUTPUT_DIR=/home/t-kaiyuangao/workspace/proj_logs/nlm_inst/inst_0621_bsz256_lr2e5_0624_step${step}_gendata
    TOKENIZER_HOME=/home/t-kaiyuangao/sfmdata-container/llama/Meta-Llama-3-8B
    mkdir -p OUTPUT_DIR

    # Execute the Python module with the specified arguments
    python3 sfm/tasks/nlm/eval/run_nlm_8b_inference.py \
        --ckpt_home "$CKPT_HOME" \
        --tokenizer_home "$TOKENIZER_HOME" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR"
done
