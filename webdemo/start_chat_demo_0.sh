#!/bin/bash

export MIXTRAL_BLOB_PATH='/home/yeqi/mount/nlm/Mixtral-8x7B-v0.1/'
export CKPT='/home/yeqi/mount/nlm/shufxi/nlm/8x7b/inst/20240903153854/global_step7755/'
export LOCAL_PATH='/data/yeqi/cache/nlm_ckpts/hf/inst_240903_s7755/'
export DEMO_MODE='chat'
export DEMO_NAME='inst_240903_s7755'
export SERVER_PORT=8234

# use GPU 0, 1
export N_MODEL_PARALLEL=2
export MODEL_START_DEVICE=0

python webdemo/main.py
