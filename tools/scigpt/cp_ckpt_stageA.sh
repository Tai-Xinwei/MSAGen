#!/bin/bash
set -euo pipefail
set -x

SRC_FOLDER=/hai1/ds_dataset/output/warmupspecialtoken/global_step25999
TGT_FOLDER=/hai1/shufxi/scigpt/7b/stageA

mkdir -p $TGT_FOLDER

cp $SRC_FOLDER/layer_00-model_states.pt $TGT_FOLDER/model.hybrid_emb.pt

# cp layer_{01,33}-model_states.pt as model.layers.{0,31}.pt
for i in {1..31}
do
    cp $SRC_FOLDER/layer_$(printf "%02d" $i)-model_states.pt $TGT_FOLDER/model.layers.$((i-1)).pt
done

cp $SRC_FOLDER/layer_32-model_states.pt $TGT_FOLDER/model.norm.pt
# 33 are dummy, skip
cp $SRC_FOLDER/layer_34-model_states.pt $TGT_FOLDER/model.lm_head.pt
