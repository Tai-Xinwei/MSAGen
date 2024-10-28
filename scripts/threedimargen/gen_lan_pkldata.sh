CKPT=/msralaphilly2/ml-la/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/checkpoint_E40.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
#INPUT=/msralaphilly2/ml-la/yinxia/wu2/backup/SFM_for_material.20240430/instruct_mat_7b_beam4_06282024.pkl
INPUT=/msralaphilly2/ml-la/yinxia/wu2/backup/SFM_for_material.20240430/instruct_mat_8b_beam4_07022024.pkl
INPUT_FNAME=$(basename $INPUT)

#OUTPUT=instruct_mat_7b_beam4_06282024
OUTPUT=instruct_mat_8b_beam4_07022024

mkdir ${OUTPUT} -p

#rm ${OUTPUT}

# if the output file already exists, ignore
python scripts/threedimargen/gen_threedimargenlan_pkldata.py \
    --dict_path sfm/data/threedimargen_data/dict_lan.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer lan \
    --infer --infer_batch_size 64 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT}
