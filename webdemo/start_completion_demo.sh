#!/bin/bash

export MIXTRAL_BLOB_PATH='/home/shufxi/nlm/Mixtral-8x7B-v0.1'
export CKPT='/home/shufxi/nlm/shufxi/nlm/8x7b/stageB_pp8_acc16_total1536_12m_bsz/global_step32000'
export LOCAL_PATH='/dev/shm/nlmoe_base'
export DEMO_MODE='completion'
export SERVER_PORT=8236

# use GPU 2, 3
export N_MODEL_PARALLEL=2
export MODEL_START_DEVICE=2

python webdemo/main.py
