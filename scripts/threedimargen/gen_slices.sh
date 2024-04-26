CKPT=/hai1/SFM/threedimargen/outputs/3dargenslices_v0.1_base_mp_nomad_qmdb_ddp_epoch50_warmup8000_lr1e-4_wd0.1_bs256/checkpoint_E2.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
INPUT=/hai1/SFM/threedimargen/data/materials_data/mp_20_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/mpts-52_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/carbon_24_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/perov_5_test.jsonl
INPUT_FNAME=$(basename $INPUT)

OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl
#rm ${OUTPUT}

# if the output file already exists, ignore
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python sfm/tasks/threedimargen/gen_threedimargenslices.py \
    --dict_path sfm/data/threedimargen_data/dict_slices.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer slices \
    --infer --infer_batch_size 128 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \

fi

python scripts/threedimargen/evaluate.py ${OUTPUT} --valid True --output ${CKPT_FOLDER}/eval.log
