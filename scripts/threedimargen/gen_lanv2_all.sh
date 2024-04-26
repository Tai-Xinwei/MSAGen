CKPT_FOLDER=/hai1/SFM/threedimargen/outputs/3dargenlanv2_v0.1_base_mp_nomad_qmdb_ddp_noniggli_epoch50_warmup8000_lr1e-4_wd0.1_bs256
CKPT_LIST=`ls ${CKPT_FOLDER}/checkpoint_E*.pt`
for CKPT in ${CKPT_LIST}; do
CKPT_NAME=$(basename $CKPT)
INPUT=/hai1/SFM/threedimargen/data/materials_data/mp_20_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/mpts-52_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/carbon_24_test.jsonl
INPUT_FNAME=$(basename $INPUT)

SG_FLAG=0
if [ $SG_FLAG -eq 1 ]; then
    OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl
    SG="--space_group"
else
    OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}_nosg.jsonl
    SG="--no_space_group"
fi

#rm ${OUTPUT}

# if the output file already exists, ignore
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python sfm/tasks/threedimargen/gen_threedimargenlan.py \
    --dict_path sfm/data/threedimargen_data/dict_lan_v2.txt \
    --tokenizer "lanv2" \
    --loadcheck_path ${CKPT} \
    --infer --infer_batch_size 128 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    ${SG}
fi

python scripts/threedimargen/evaluate.py ${OUTPUT} --valid True --output ${CKPT_FOLDER}/eval.log
done
