CKPT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_ft_mp_20_ddp_noniggli_base_epoch10_warmup1_lr1e-5_wd0.1_bs16/checkpoint_E9.pt
#INPUT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/instruct/base1b/instruct_task_20240807/1b_dialogue_1v1_bs2048_steps_20000/all/test.bandgap_to_mat.tsv.response.valid.txt
#OUTPUT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/instruct/base1b/instruct_task_20240807/1b_dialogue_1v1_bs2048_steps_20000/all/test.bandgap_to_mat.response.valid.structures
#INPUT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/instruct/base8b/instruct_task_20240807/8b_dialogue_1v1_bs2048_steps_20000/all/test.bandgap_to_mat.tsv.response.valid.txt
#OUTPUT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/instruct/base8b/instruct_task_20240807/8b_dialogue_1v1_bs2048_steps_20000/all/test.bandgap_to_mat.response.valid.structures
#INPUT=/sfmdataeastus2/nlm/zekun/instruct/base8b/uncondition_summary_1w_1w/1/material.valid.txt
#OUTPUT=/sfmdataeastus2//nlm/zekun/instruct/base8b/uncondition_summary_1w_1w/1/material.valid.structures
INPUT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/instruct/8x7b/global_step7755_restore_new/test.bulk.response.valid.txt
OUTPUT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/data/materials_data/instruct/8x7b/global_step7755_restore_new/test.bulk.response.valid.structures

mkdir ${OUTPUT} -p

# if the output file already exists, ignore
python scripts/threedimargen/gen_threedimargenlan_txtdata.py \
    --dict_path sfm/data/threedimargen_data/dict_lan.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer lan \
    --infer --infer_batch_size 64 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT}
