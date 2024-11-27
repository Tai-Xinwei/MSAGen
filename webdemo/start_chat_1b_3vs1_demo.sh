#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MODEL_TYPE='llama1b'
export LLAMA_BLOB_PATH='/home/shufxi/nlm/llama/Meta-Llama-3-8B/original'
export CKPT='/home/shufxi/nlm/zekun/output/1b/SFMMolInstruct.20240807_v2_dialogue_3vs1_bs2048/global_step20000'
export DEMO_MODE='completion' #'chat'
export SERVER_PORT=8238

python webdemo/main.py
