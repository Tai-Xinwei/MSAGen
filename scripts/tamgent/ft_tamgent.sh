set -x
set -euo pipefail

ulimit -c unlimited


export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

echo "Training"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    sfm/tasks/tamgent/finetune_tamgent2.py \
    --init_lr 3e-5 \
    --min_lr 8e-6 \
    --warmup_lr 1e-6 \
    --weight_decay 0.05 \
    --total_num_epochs 25 \
    --warmup_num_steps 5000 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --val_batch_size 1 \
    --save_dir $AMLT_OUTPUT_DIR \
    --log_interval 100 \
    --strategy DDP \
    --fp16
