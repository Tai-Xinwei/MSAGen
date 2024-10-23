#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export MODEL_TYPE='llama8b'
export LLAMA_BLOB_PATH='/home/shufxi/nlm/llama/Meta-Llama-3-8B/original'
export CKPT='/home/shufxi/nlm/zekun/output/base8b/SFMMolInstruct.20240807_v2_dialogue_1vs1_bs2048/global_step20000'
export DEMO_MODE='chat' #'completion' #'chat'
export SERVER_PORT=8238

python webdemo/main.py
