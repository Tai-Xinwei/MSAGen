#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# for a checkpoint in storage account
python sfm/tasks/generalist/test_generalist.py \
    --num_gpus 8 \
    --test_checkpoint_path remote:checkpoints/llama2-7b-mol-desc-func-group-adaptor/ \
    --test_global_step 2499 \
    --remote_checkpoint_query_string "<the SAS query string>" \
    --remote_checkpoint_storage_account hai1data \
    --remote_checkpoint_storage_container mfm \
    --local_checkpoint_path /mnt/shiyu/models/converted/ \
    --question_list_fname question.txt \
    --smiles_list_fname smiles.txt \
    --llm_model_name_or_path /mnt/shiyu/models/converted/llama-2-7b/ \
    --infer_batch_size 1 \
    --output_fname output.txt \
    --ft


# for a local checkpoint
# python sfm/tasks/generalist/test_generalist.py \
#     --num_gpus 8 \
#     --test_checkpoint_path remote:checkpoints/llama2-7b-mol-desc-func-group-adaptor/ \
#     --test_global_step 2499 \
#     --remote_checkpoint_query_string "" \
#     --remote_checkpoint_storage_account "" \
#     --remote_checkpoint_storage_container "" \
#     --local_checkpoint_path "" \
#     --question_list_fname question.txt \
#     --smiles_list_fname smiles.txt \
#     --llm_model_name_or_path /mnt/shiyu/models/converted/llama-2-7b/ \
#     --infer_batch_size 1 \
#     --output_fname output2.txt \
#     --ft
