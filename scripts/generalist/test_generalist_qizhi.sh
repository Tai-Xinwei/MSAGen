#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# for a checkpoint in storage account
# python sfm/tasks/generalist/test_generalist.py \
#     --num_gpus 8 \
#     --test_checkpoint_path remote:checkpoints/llama2-7b-mol-desc-func-group-adaptor/ \
#     --test_global_step 2499 \
#     --remote_checkpoint_query_string "<the SAS query string>" \
#     --remote_checkpoint_storage_account hai1data \
#     --remote_checkpoint_storage_container mfm \
#     --local_checkpoint_path /mnt/shiyu/models/converted/ \
#     --question_list_fname question.txt \
#     --smiles_list_fname smiles.txt \
#     --llm_model_name_or_path /mnt/shiyu/models/converted/llama-2-7b/ \
#     --infer_batch_size 1 \
#     --output_fname output.txt \
#     --ft


# for a local checkpoint
# python sfm/tasks/generalist/test_generalist_pp.py \

# torchrun sfm/tasks/generalist/test_generalist_pp.py \
#     --test_checkpoint_path /home/v-peiqizhi/SFM_framework/output/ESOL_training_0815/global_step13999 \
#     --data_path /sfm/ds_dataset/qizhi_numerical/ESOL \
#     --batch_size 4 \
#     --seed 42 \
#     --dataset_names mol-instruction-mol-desc \
#     --dataset_splits test \
#     --ft \
#     --pipeline_model_parallel_size 1 \
#     --strategy Pipeline \
#     --sandwich_ln \
#     --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b

# torchrun sfm/tasks/generalist/test_generalist_pp.py \
#     --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/freesolv_training_0824/global_step5999 \
#     --data_path /sfm/ds_dataset/qizhi_numerical/freesolv \
#     --batch_size 4 \
#     --seed 42 \
#     --dataset_names mol-instruction-mol-desc \
#     --dataset_splits test \
#     --ft \
#     --pipeline_model_parallel_size 1 \
#     --strategy Pipeline \
#     --sandwich_ln \
#     --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b

# for step in 499 1499 2499 3499 4499 5499 6499 7499 8499 9499 10499 11499 12499 13499 14499
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/bbbp_training_0824/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/bbbp \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a bbbp.log
# done

# for step in 499 1499 2499 3499 4499 5499 6499 7499 8499 9499 10499 11499 12499 13499 14499
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/bace_training_0824/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/bace \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_bace.log
# done


# torchrun sfm/tasks/generalist/test_generalist_pp.py \
#     --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/bace_training_0824/global_step14499 \
#     --data_path /sfm/ds_dataset/qizhi_numerical/bace_re \
#     --batch_size 4 \
#     --seed 42 \
#     --dataset_names mol-instruction-mol-desc \
#     --dataset_splits test \
#     --ft \
#     --pipeline_model_parallel_size 1 \
#     --strategy Pipeline \
#     --sandwich_ln \
#     --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_bace_re.log

# for((step=999; step<=40999; step+=1000));
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/lipo_training_0826/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/lipo \
#         --test_task reg \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_lipo_long.log
# done

# for((step=999; step<=40999; step+=1000));
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/hiv_training_0825/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/hiv \
#         --test_task cls \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_hiv.log
# done

# for((step=999; step<=13999; step+=1000));
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/ESOL_training_0815/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/ESOL \
#         --test_task reg \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_esol.log
# done

# for((step=999; step<=7999; step+=500));
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/freesolv_training_0824/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/freesolv \
#         --test_task reg \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_freesolv.log
# done

# for((step=999; step<=17999; step+=1000));
# do
#     torchrun sfm/tasks/generalist/test_generalist_pp.py \
#         --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/bbbp_training_mlp_0828/global_step${step} \
#         --data_path /sfm/ds_dataset/qizhi_numerical/bbbp \
#         --test_task cls_mlp \
#         --batch_size 4 \
#         --seed 42 \
#         --dataset_names mol-instruction-mol-desc \
#         --dataset_splits test \
#         --ft \
#         --pipeline_model_parallel_size 1 \
#         --strategy Pipeline \
#         --sandwich_ln \
#         --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_bbbp_mlp.log
# done

for((step=999; step<=16999; step+=1000));
do
    for test_data in ESOL freesolv lipo
    do
        torchrun sfm/tasks/generalist/test_generalist_pp.py \
            --test_checkpoint_path /sfm/ds_dataset/output/qizhi_ft/molnet_3_reg_0829/global_step${step} \
            --data_path /sfm/ds_dataset/qizhi_numerical/${test_data} \
            --test_task reg \
            --batch_size 4 \
            --seed 42 \
            --dataset_names mol-instruction-mol-desc \
            --dataset_splits test \
            --ft \
            --pipeline_model_parallel_size 1 \
            --strategy Pipeline \
            --sandwich_ln \
            --llm_model_name_or_path /home/v-peiqizhi/models/converted/llama-2-7b | tee -a test_molnet_3_reg.log
        echo "Done for test_data: ${test_data}, step: ${step}"
    done
done

# Check sandwich_ln ??
