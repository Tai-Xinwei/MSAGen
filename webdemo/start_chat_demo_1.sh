#!/bin/bash

export MIXTRAL_BLOB_PATH='/home/yeqi/mount/nlm/Mixtral-8x7B-v0.1/'
export CKPT='None'
export LOCAL_PATH='/data/yeqi/cache/nlm_ckpts/hf/inst_mix500_s1717/nlm_moe/'
export DEMO_MODE='chat'
export DEMO_NAME='inst_mix500_s1717'
export SERVER_PORT=8238

# use GPU 3, 4
export N_MODEL_PARALLEL=2
export MODEL_START_DEVICE=2

python webdemo/main.py
