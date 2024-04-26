CKPT=/hai1/SFM/threedimargen/outputs/3darenergy_v0.5_2500k_ds10_es-1e-2_ddp_layer12_head16_epoch100_warmup8000_lr1e-4_bs256/checkpoint_E0.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
INPUT=/hai1/SFM/threedimargen/data/materials_data/coredataset-v20230731_test.jsonl
#INPUT=/hai1/SFM/threedimargen/data/materials_data/core_tmp.jsonl
INPUT_FNAME=$(basename $INPUT)
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

rm ${OUTPUT}

# if the output file already exists, ignore
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python sfm/tasks/threedimargen/gen_threedimargennumenergy.py \
    --dict_path sfm/data/threedimargen_data/dict.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 128 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT}
fi

python scripts/threedimargen/evaluate_energy.py ${OUTPUT}
