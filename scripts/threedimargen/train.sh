#python
torchrun --nnodes 1 --nproc_per_node 1 sfm/tasks/threedimargen/train_threedimargen.py \
    --save_dir /hai1/SFM/threedimargen/outputs/3dargen_v0.4_mp_nomad_dedup_scal10_zero1_layer24_head16_epoch100_warmup8000_lr1e-4_bs32 \
    --train_data_path /hai1/SFM/threedimargen/data/materials_data/mp_nomad_dedup_train.jsonl \
    --valid_data_path /hai1/SFM/threedimargen/data/materials_data/mp_nomad_dedup_valid.jsonl \
    --fp16 \
    --total_num_epochs 100 \
    --warmup_num_steps 8000 \
    --max_lr 1e-4 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --log_interval 100 \
    --strategy Zero1 \
    --scale_digit 10
