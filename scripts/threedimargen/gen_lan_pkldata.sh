CKPT=/hai1/renqian/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/checkpoint_E40.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
INPUT=/hai1/renqian/SFM/threedimargen/data/materials_data/instructv1_mat_sample.unique.smactvalid.pkl
INPUT_FNAME=$(basename $INPUT)

OUTPUT=${CKPT_FOLDER}/instructv1_mat_sample/

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
