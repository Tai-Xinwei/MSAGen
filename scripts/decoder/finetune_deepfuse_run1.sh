set -x
set -euo pipefail

ulimit -c unlimited


export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

echo "Training"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

BLOB=${1:-/blob}

deepspeed --num_gpu=$NUM_GPUS --master_port=12345 \
    sfm/tasks/decoder/train_dec_deepfuse.py \
    --strategy Pipeline --pipeline_model_parallel_size 4 \
    --pp_partition_layer_name manual  \
    --pp_part_list '[0, 9, 17, 25, 34]' \
    --init_lr 3e-5 \
    --min_lr 8e-6 \
    --warmup_lr 1e-6 \
    --weight_decay 0.05 \
    --total_num_epochs 25 \
    --warmup_num_steps 5000 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --val_batch_size 8 \
    --save_dir $AMLT_OUTPUT_DIR \
    --log_interval 20 \
    --fp16 \
    --llama_model $BLOB/ds_dataset/llama2/llama-2-7b \
    --entity_decoder_model  $BLOB/shufxi/mixgpt_new/ckpt \
    --train_mol_path $BLOB/shufxi/data/tamgent/chebi/train.textmol.smi \
    --train_text_path $BLOB/shufxi/data/tamgent/chebi/train.textmol.desc \
    --val_mol_path $BLOB/shufxi/data/tamgent/chebi/val.textmol.smi \
    --val_text_path $BLOB/shufxi/data/tamgent/chebi/val.textmol.desc \
