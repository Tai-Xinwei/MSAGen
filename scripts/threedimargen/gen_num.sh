CKPT=/hai1/SFM/threedimargen/outputs/3dargen_v0.6_mp_nomad_qmdb_scal10_ddp_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.0_bs256/checkpoint_E40.pt
#CKPT=/hai1/SFM/threedimargen/outputs/3dargen_v0.5_mp_nomad_qmdb_scal10_reorder_ddp_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.0_bs256/checkpoint_E49.pt
#CKPT=/hai1/SFM/threedimargen/outputs/3dargen_v0.5_mp_nomad_qmdb_scal10_ddp_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.0_bs256/checkpoint_E49.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
INPUT=/hai1/SFM/threedimargen/data/materials_data/mp_20_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/mpts-52_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/carbon_24_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/perov_5_test.jsonl
INPUT_FNAME=$(basename $INPUT)
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}_nosg.jsonl

rm ${OUTPUT}

# if the output file already exists, ignore
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python sfm/tasks/threedimargen/gen_threedimargennum.py \
    --dict_path sfm/data/threedimargen_data/dict.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 128 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --no_space_group
fi

python scripts/threedimargen/evaluate.py ${OUTPUT}
