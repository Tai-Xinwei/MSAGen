#!/bin/bash

export MIXTRAL_BLOB_PATH='/home/shufxi/nlm/Mixtral-8x7B-v0.1'
export CKPT='/home/shufxi/nlm/shufxi/nlm/8x7b/inst/20240903153854/global_step7755'
export LOCAL_PATH='/dev/shm/nlmoe'
export DEMO_MODE='chat'
export SERVER_PORT=8234

# use GPU 0, 1
export N_MODEL_PARALLEL=2
export MODEL_START_DEVICE=0

python webdemo/main.py
