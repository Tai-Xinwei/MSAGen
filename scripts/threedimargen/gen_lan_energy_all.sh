CKPT=/hai1/SFM/threedimargen/outputs/3dargenlanenergy_v0.1_base_2500k_ddp_epoch50_warmup8000_lr1e-4_bs256/checkpoint_E45.pt
CKPT=/hai1/SFM/threedimargen/outputs/3dargenlanv2energy_v0.1_100m_2500k_ddp_epoch50_warmup8000_lr1e-4_bs256/checkpoint_E46.pt
CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)
INPUT_FOLDER=/hai1/SFM/threedimargen/data/materials_data/energy
INPUT_FNAMES=$(find ${INPUT_FOLDER} -name "*.jsonl" -type f)

for INPUT in ${INPUT_FNAMES}; do
    INPUT_FNAME=$(basename $INPUT)
    OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

    rm ${OUTPUT}

    # if the output file already exists, ignore
    if [ -f ${OUTPUT} ]; then
        echo "Output file ${OUTPUT} already exists. Skipping."
    else
        python sfm/tasks/threedimargen/gen_threedimargenlanenergy.py \
        --dict_path sfm/data/threedimargen_data/dict_lan.txt \
        --loadcheck_path ${CKPT} \
        --infer --infer_batch_size 64 \
        --input_file ${INPUT} \
        --output_file ${OUTPUT}
    fi

    echo ${INPUT_FNAME}
    python scripts/threedimargen/evaluate_energy.py ${OUTPUT}
done
