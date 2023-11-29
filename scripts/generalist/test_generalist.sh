#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# for a checkpoint in storage account, run with 8 GPUs, data parallal evaluation
torchrun --nproc_per_node 8 sfm/tasks/generalist/test_generalist.py \
    --num_gpus 8 \
    --test_checkpoint_path remote:generalist-checkpoints/ft_100MMFM_7Bppllama_graphqformer1_moldesc+funcgroup+funcgroup-desc+chebi+pubchem-adaptor-special-tokens-2e-4-pp16-layerwise-graph-attn-bias/ \
    --test_global_step 13999 \
    --remote_checkpoint_query_string "<SAS>" \
    --remote_checkpoint_storage_account hai1data \
    --remote_checkpoint_storage_container mfm \
    --local_checkpoint_path /mnt/shiyu/models/converted/ \
    --question_list_fname sfm/tasks/generalist/outputs/question.txt \
    --smiles_list_fname sfm/tasks/generalist/outputs/smiles.txt \
    --llm_model_name_or_path /mnt/shiyu/models/converted/llama-2-7b/ \
    --infer_batch_size 1 \
    --output_fname output.txt \
    --ft \
    --fused_graphormer_llama \
    --add_mol_attn_bias_in_llama \
    --path_edge_cutoff 5 \
    --eval_method AROMATIC \
    --num_eval_repeats 10

# for a checkpoint in storage account, run with only 1 GPU
# python sfm/tasks/generalist/test_generalist.py \
#     --num_gpus 1 \
#     --test_checkpoint_path remote:generalist-checkpoints/ft_100MMFM_7Bppllama_graphqformer1_moldesc+funcgroup+funcgroup-desc+chebi+pubchem-adaptor-special-tokens-2e-4-pp16-layerwise-graph-attn-bias/ \
#     --test_global_step 13999 \
#     --remote_checkpoint_query_string "<SAS>" \
#     --remote_checkpoint_storage_account hai1data \
#     --remote_checkpoint_storage_container mfm \
#     --local_checkpoint_path /mnt/shiyu/models/converted/ \
#     --question_list_fname sfm/tasks/generalist/outputs/question.txt \
#     --smiles_list_fname sfm/tasks/generalist/outputs/smiles.txt \
#     --llm_model_name_or_path /mnt/shiyu/models/converted/llama-2-7b/ \
#     --infer_batch_size 1 \
#     --output_fname output.txt \
#     --ft \
#     --fused_graphormer_llama \
#     --add_mol_attn_bias_in_llama \
#     --path_edge_cutoff 5 \
#     --eval_method AROMATIC \
#     --num_eval_repeats 10
