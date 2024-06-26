#!/bin/bash

for step in 54000
do
    CKPT_HOME=/home/t-kaiyuangao/sfmdata-container/kaiyuan/results/nlm/inst/inst_tuning_full_bsz128_lr5e-5_0616/global_step${step}
    INPUT_DIR=/home/t-kaiyuangao/ml-container/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240524.test/fulltest
    OUTPUT_DIR=/home/t-kaiyuangao/workspace/proj_logs/nlm_inst/inst_tuning_full_bsz128_lr5e-5_0616_step${step}
    mkdir -p OUTPUT_DIR

    # Execute the Python module with the specified arguments
    python3 run_nlm_inference.py \
        --ckpt_home "$CKPT_HOME" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR"
done
