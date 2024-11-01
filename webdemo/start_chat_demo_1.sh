#!/bin/bash

# export MODEL_TYPE="ph3_mini"
export MIXTRAL_BLOB_PATH='/home/yeqi/mount/nlm/Mixtral-8x7B-v0.1/'
export LLAMA_BLOB_PATH='/home/shufxi/nlm/llama1b/inst/20240705132320/global_step44960/'
export DEP_CKPT='/home/yeqi/mount/nlm/shufxi/nlm/8x7b/inst/20240903153854/global_step7755/'
export CKPT='None'
export LOCAL_PATH='/data/yeqi/cache/nlm_ckpts/hf/inst_240903_s7755/global_step7755_restore/nlmoe'
export DEMO_MODE='chat_dev'
export DEMO_NAME='inst_240903_s7755'
export SERVER_PORT=8238

# use GPU 3, 4
export N_MODEL_PARALLEL=2
export MODEL_START_DEVICE=2

python webdemo/main.py
