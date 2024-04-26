CKPT=/hai1/SFM/threedimargen/outputs/3darenergylan_v0.1_100m_2500k_ddp_epoch50_warmup8000_lr1e-4_bs256/checkpoint_E9.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
INPUT=/hai1/SFM/threedimargen/data/materials_data/coredataset-v20230731_test.jsonl
INPUT_FNAME=$(basename $INPUT)
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

#rm ${OUTPUT}

# if the output file already exists, ignore
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python sfm/tasks/threedimargen/gen_threedimargenlanenergy.py \
    --dict_path sfm/data/threedimargen_data/dict_lan.txt \
    --loadcheck_path ${CKPT} \
    --infer --infer_batch_size 128 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT}
fi

python scripts/threedimargen/evaluate_energy.py ${OUTPUT}
