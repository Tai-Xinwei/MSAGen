#!/bin/bash

python -u sfm/tasks/scigpt/bp_gen.py \
    --ckpt_home '/blob/shufxi/scigpt/7bv2/stageB/finetune/bpE5/global_step110' \
    --tokenizer_home '/hai1/ds_dataset/llama2/llama-2-7b' \
    --output_path '/blob/shufxi/scigpt/7bv2/stageB/finetune/bpE5/bp_gen.txt'

echo "Done"
